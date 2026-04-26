import json
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from datetime import datetime

from flask import Flask, Response, render_template, request, send_file

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

app = Flask(__name__)
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()

TIMESTAMP_RE = re.compile(r'\[(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->')

WHISPER_EXE = os.path.join(BASE_DIR, "whisper.cpp-windows-vulkan", "whisper-cli.exe")
MODEL_PATH  = os.path.join(BASE_DIR, "whisper.cpp-windows-vulkan", "ggml-large-v3-turbo-q5_0.bin")
FFPROBE_EXE = os.path.join(BASE_DIR, "ffprobe.exe")
AUDIO_DIR   = os.environ.get("AUDIO_DIR", BASE_DIR)
AUDIO_EXT   = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus", ".wma", ".mp4", ".webm"}

DEFAULT_PROMPT = """\
YouTube 영상의 전사 텍스트를 구조화된 요약본으로 재작성해줘.
---
전사 텍스트:
{transcript}
"""

PROMPT_FILE   = os.path.join(BASE_DIR, "prompt.txt")
GEMINI_MODEL  = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


# ── Filename helpers ──────────────────────────────────────────────────

def ts_prefix() -> str:
    """Return YYYYMMDDHHmm_ timestamp prefix."""
    return datetime.now().strftime("%Y%m%d%H%M_")


def sanitize(title: str) -> str:
    s = re.sub(r'[\\/:*?"<>|\r\n\t]', '_', title).strip().strip('.')
    return (s or "audio")[:80]


def unique_path(directory: str, stem: str, ext: str) -> str:
    path = os.path.join(directory, stem + ext)
    if not os.path.exists(path):
        return path
    i = 1
    while True:
        path = os.path.join(directory, f"{stem}_{i}{ext}")
        if not os.path.exists(path):
            return path
        i += 1


# ── Duration / progress helpers ───────────────────────────────────────

def get_file_duration(file_path: str) -> float:
    if not os.path.exists(FFPROBE_EXE):
        return 0.0
    try:
        r = subprocess.run(
            [FFPROBE_EXE, "-v", "quiet", "-print_format", "json", "-show_format", file_path],
            capture_output=True, text=True, timeout=15, encoding="utf-8",
        )
        data = json.loads(r.stdout)
        return float(data.get("format", {}).get("duration") or 0)
    except Exception:
        return 0.0


def _format_duration(secs: float) -> str:
    h, rem = divmod(int(secs), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _emit_progress(q: queue.Queue, line: str, total: float) -> None:
    m = TIMESTAMP_RE.search(line)
    if m and total > 0:
        current = (int(m.group(1)) * 3600 + int(m.group(2)) * 60
                   + int(m.group(3)) + int(m.group(4)) / 1000)
        q.put({"type": "progress", "pct": min(99, int(current / total * 100))})


def _stage(q: queue.Queue, stage: str) -> None:
    q.put({"type": "stage", "stage": stage})


# ── Whisper post-processing ───────────────────────────────────────────

def _parse_and_save(json_path: str, txt_path: str) -> str | None:
    if not os.path.exists(json_path):
        return None
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("transcription", [])
    cleaned, prev = [], ""
    for seg in segments:
        text = seg.get("text", "").strip()
        if text and text != prev:
            cleaned.append(text)
            prev = text
    transcript = "\n".join(cleaned)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    os.remove(json_path)
    return transcript


def _run_whisper(job_id: str, audio_path: str, language: str,
                 threads: str, total: float) -> int:
    q = jobs[job_id]["queue"]
    stop = jobs[job_id]["stop_event"]
    cmd = [
        WHISPER_EXE, "-m", MODEL_PATH, "-f", audio_path,
        "--language", language, "--threads", threads,
        "--output-json", "--temperature", "0",
        "--best-of", "5", "--no-speech-thold", "0.8",
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, cwd=BASE_DIR,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"}, encoding="utf-8",
    )
    jobs[job_id]["proc"] = proc

    for line in proc.stdout:
        if stop.is_set():
            proc.kill()
            break
        line = line.rstrip("\n")
        q.put(line)
        _emit_progress(q, line, total)

    proc.wait()
    jobs[job_id]["proc"] = None
    return proc.returncode


# ── Job runners ───────────────────────────────────────────────────────

def run_job(job_id: str, params: dict) -> None:
    q    = jobs[job_id]["queue"]
    stop = jobs[job_id]["stop_event"]
    url  = params["url"]
    lang = params.get("language", "auto")
    thr  = str(params.get("threads", 6))

    # 1. Fetch video info
    q.put("영상 정보 가져오는 중...")
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(url, download=False)
        title = info.get("title", "audio")
        total = float(info.get("duration") or 0)
    except Exception as e:
        q.put(f"오류: {e}")
        jobs[job_id]["status"] = "error"
        q.put(None)
        return

    jobs[job_id]["total_duration"] = total
    if total > 0:
        q.put(f"영상 길이: {_format_duration(total)}")
        q.put({"type": "duration", "seconds": total})

    # 2. Download
    os.makedirs(AUDIO_DIR, exist_ok=True)
    prefix     = ts_prefix()
    stem       = prefix + sanitize(title)
    audio_path = unique_path(AUDIO_DIR, stem, ".mp3")
    stem_final = os.path.splitext(os.path.basename(audio_path))[0]

    _stage(q, "download")
    q.put(f"다운로드 중: {title}")

    def _progress_hook(d):
        if stop.is_set():
            raise Exception("사용자가 취소함")
        if d["status"] == "downloading":
            pct   = d.get("_percent_str", "").strip()
            speed = d.get("_speed_str", "").strip()
            q.put(f"  다운로드: {pct}  {speed}")
        elif d["status"] == "finished":
            q.put("다운로드 완료")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.splitext(audio_path)[0],
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
        "quiet": True,
        "progress_hooks": [_progress_hook],
    }
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        if stop.is_set():
            jobs[job_id]["status"] = "cancelled"
        else:
            q.put(f"다운로드 오류: {e}")
            jobs[job_id]["status"] = "error"
        q.put(None)
        return

    if stop.is_set() or not os.path.exists(audio_path):
        jobs[job_id]["status"] = "cancelled" if stop.is_set() else "error"
        q.put(None)
        return

    q.put(f"저장됨: {os.path.basename(audio_path)}")

    # 3. Transcribe
    _stage(q, "transcribe")
    q.put("Whisper 전사 시작...")
    rc = _run_whisper(job_id, audio_path, lang, thr, total)

    if stop.is_set():
        jobs[job_id]["status"] = "cancelled"
        q.put("중지됨.")
        q.put(None)
        return

    # 4. Parse → save txt
    json_path = audio_path + ".json"
    txt_path  = unique_path(AUDIO_DIR, stem_final, ".txt")

    if rc == 0:
        transcript = _parse_and_save(json_path, txt_path)
        if transcript is not None:
            q.put(f"✅ 전사 저장됨: {os.path.basename(txt_path)}")
            jobs[job_id].update({"status": "done", "result": transcript, "output_file": txt_path})
            q.put({"type": "progress", "pct": 100})
            q.put(None)
            return
    jobs[job_id]["status"] = "error"
    q.put(None)


def run_file_job(job_id: str, file_path: str, params: dict) -> None:
    q    = jobs[job_id]["queue"]
    stop = jobs[job_id]["stop_event"]
    lang = params.get("language", "auto")
    thr  = str(params.get("threads", 6))

    q.put(f"파일: {os.path.basename(file_path)}")

    total = get_file_duration(file_path)
    jobs[job_id]["total_duration"] = total
    if total > 0:
        q.put(f"파일 길이: {_format_duration(total)}")
        q.put({"type": "duration", "seconds": total})

    _stage(q, "transcribe")
    q.put("Whisper 전사 시작...")
    rc = _run_whisper(job_id, file_path, lang, thr, total)

    if stop.is_set():
        jobs[job_id]["status"] = "cancelled"
        q.put("중지됨.")
        q.put(None)
        return

    audio_dir = os.path.dirname(file_path)
    stem      = ts_prefix() + os.path.splitext(os.path.basename(file_path))[0]
    json_path = file_path + ".json"
    txt_path  = unique_path(audio_dir, stem, ".txt")

    if rc == 0:
        transcript = _parse_and_save(json_path, txt_path)
        if transcript is not None:
            q.put(f"✅ 전사 저장됨: {os.path.basename(txt_path)}")
            jobs[job_id].update({"status": "done", "result": transcript, "output_file": txt_path})
            q.put({"type": "progress", "pct": 100})
            q.put(None)
            return
    jobs[job_id]["status"] = "error"
    q.put(None)


# ── Routes ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", default_prompt=DEFAULT_PROMPT)


@app.route("/files")
def list_files():
    files = []
    try:
        for fname in os.listdir(AUDIO_DIR):
            if os.path.splitext(fname)[1].lower() not in AUDIO_EXT:
                continue
            path = os.path.join(AUDIO_DIR, fname)
            if not os.path.isfile(path):
                continue
            files.append({
                "name": fname,
                "path": path,
                "size": os.path.getsize(path),
                "duration": get_file_duration(path),
            })
    except Exception as e:
        return json.dumps({"error": str(e)}), 500
    files.sort(key=lambda x: x["name"].lower())
    return json.dumps({"files": files, "dir": AUDIO_DIR})


@app.route("/start", methods=["POST"])
def start():
    data   = request.get_json(force=True)
    source = data.get("source", "url")

    if source == "file":
        file_path = (data.get("file_path") or "").strip()
        if not file_path or not os.path.exists(file_path):
            return json.dumps({"error": "파일을 찾을 수 없습니다."}), 400
    else:
        url = (data.get("url") or "").strip()
        if not url:
            return json.dumps({"error": "URL을 입력해주세요."}), 400

    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {
            "status": "running",
            "queue":  queue.Queue(),
            "result": None,
            "output_file": None,
            "start_time":  time.time(),
            "total_duration": 0,
            "proc": None,
            "stop_event": threading.Event(),
        }

    if source == "file":
        t = threading.Thread(target=run_file_job, args=(job_id, file_path, data), daemon=True)
    else:
        t = threading.Thread(target=run_job, args=(job_id, data), daemon=True)
    t.start()
    return json.dumps({"job_id": job_id})


@app.route("/stop/<job_id>", methods=["POST"])
def stop_job(job_id: str):
    if job_id not in jobs:
        return json.dumps({"error": "Not found"}), 404
    job = jobs[job_id]
    job["stop_event"].set()
    proc = job.get("proc")
    if proc and proc.poll() is None:
        proc.kill()
    return json.dumps({"ok": True})


@app.route("/stream/<job_id>")
def stream(job_id: str):
    if job_id not in jobs:
        return "Not found", 404

    def generate():
        q = jobs[job_id]["queue"]
        while True:
            try:
                item = q.get(timeout=60)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            if item is None:
                yield "event: done\ndata: \n\n"
                break
            if isinstance(item, dict):
                yield f"event: {item['type']}\ndata: {json.dumps(item)}\n\n"
            else:
                yield f"data: {str(item).replace(chr(10), ' ')}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/result/<job_id>")
def result(job_id: str):
    if job_id not in jobs:
        return json.dumps({"error": "Not found"}), 404
    job = jobs[job_id]
    filename = os.path.basename(job["output_file"]) if job["output_file"] else None
    return json.dumps({"status": job["status"], "result": job["result"], "filename": filename})


@app.route("/download/<job_id>")
def download(job_id: str):
    if job_id not in jobs:
        return "Not found", 404
    path = jobs[job_id].get("output_file")
    if not path or not os.path.exists(path):
        return "File not found", 404
    return send_file(path, as_attachment=True)


@app.route("/summarize/<job_id>", methods=["POST"])
def summarize(job_id: str):
    if job_id not in jobs:
        return json.dumps({"error": "Not found"}), 404

    data = request.get_json(force=True)
    prompt_template = data.get("prompt") or DEFAULT_PROMPT
    transcript = jobs[job_id].get("result")
    if not transcript:
        return json.dumps({"error": "전사 결과가 없습니다."}), 400

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        def err():
            yield f"event: error\ndata: {json.dumps('GEMINI_API_KEY가 .env 파일에 없습니다.')}\n\n"
        return Response(err(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})

    user_message = prompt_template.replace("{transcript}", transcript)

    def generate():
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(user_message, stream=True)
            for chunk in response:
                if chunk.text:
                    yield f"data: {json.dumps(chunk.text)}\n\n"
            yield "event: done\ndata: \n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps(str(e))}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/prompt", methods=["GET"])
def get_prompt():
    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, encoding="utf-8") as f:
            return json.dumps({"prompt": f.read()})
    return json.dumps({"prompt": DEFAULT_PROMPT})


@app.route("/prompt", methods=["POST"])
def save_prompt():
    data = request.get_json(force=True)
    with open(PROMPT_FILE, "w", encoding="utf-8") as f:
        f.write(data.get("prompt", ""))
    return json.dumps({"ok": True})


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000, threaded=True)
