"""YouTube 영상 전사 + 화자 분리"""

import argparse
import os
import sys
import json
import subprocess
import warnings
from datetime import datetime
from pathlib import Path

# pyannote import 전에 warning 필터 등록 (import 시점 경고 억제)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["PYANNOTE_AUDIO_NO_TORCHCODEC"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import yt_dlp
import torch
import soundfile as sf
from pyannote.audio import Pipeline

try:
    from dotenv import load_dotenv, find_dotenv
except ImportError:
    sys.exit("❌ pip install python-dotenv 필요")

# .env 로드: DOTENV_PATH > 스크립트 옆 .env > cwd 자동 탐색
_explicit = os.environ.get("DOTENV_PATH")
_script_env = Path(__file__).resolve().parent / ".env"
if _explicit and Path(_explicit).exists():
    load_dotenv(_explicit, override=False)
elif _script_env.exists():
    load_dotenv(_script_env, override=False)
else:
    load_dotenv(find_dotenv(usecwd=True), override=False)

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# HuggingFace Hub 라이브러리가 참조하는 환경변수에도 주입
if HF_TOKEN:
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", HF_TOKEN)
    os.environ.setdefault("HF_HUB_TOKEN", HF_TOKEN)

WHISPER_DIR = r"D:\PYTHON\youtube-script\whisper.cpp-windows-vulkan"
WHISPER_EXE = os.path.join(WHISPER_DIR, "whisper-cli.exe")
MODEL_PATH = os.path.join(WHISPER_DIR, "ggml-large-v3-turbo-q5_0.bin")
OUTPUT_DIR = r"D:\PYTHON\youtube-script\whisper-output"


def download_audio(url, base_name):
    """YouTube → 16kHz mono WAV"""
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': base_name + '.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '0',
        }],
        'postprocessor_args': ['-ar', '16000', '-ac', '1'],
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "unknown")

    audio_file = base_name + ".wav"
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"오디오 파일 없음: {audio_file}")
    return audio_file, title


def run_whisper(audio_file, language, threads):
    """whisper.cpp 실행 → JSON 결과"""
    output_prefix = os.path.splitext(audio_file)[0]
    cmd = [
        WHISPER_EXE, "-m", MODEL_PATH, "-f", audio_file,
        "--language", language, "--threads", str(threads),
        "--output-json", "-of", output_prefix,
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        print(result.stderr[-1000:] if result.stderr else "")
        raise RuntimeError(f"whisper-cli 실패 (code={result.returncode})")

    for path in (output_prefix + ".json", audio_file + ".json"):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f), path
    raise FileNotFoundError("Whisper JSON 출력 파일 없음")


def load_audio_for_pyannote(audio_file):
    """soundfile 로드 → pyannote 입력 dict (mono, channel-first)"""
    waveform, sample_rate = sf.read(audio_file, dtype='float32')
    waveform = torch.from_numpy(waveform).float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
    return {"waveform": waveform, "sample_rate": sample_rate}


def assign_speakers(segments, diarization):
    """세그먼트별 화자 라벨 매핑 (누적 overlap 기준)"""
    diar_turns = [(t.start, t.end, spk)
                  for t, _, spk in diarization.itertracks(yield_label=True)]
    out = []
    for seg in segments:
        s, e = seg["offsets"]["from"] / 1000.0, seg["offsets"]["to"] / 1000.0
        text = seg["text"].strip()
        if not text:
            continue
        overlaps = {}
        for ts, te, spk in diar_turns:
            ov = max(0.0, min(e, te) - max(s, ts))
            if ov > 0:
                overlaps[spk] = overlaps.get(spk, 0.0) + ov
        speaker = max(overlaps, key=overlaps.get) if overlaps else "UNKNOWN"
        out.append({"start": s, "end": e, "speaker": speaker, "text": text})
    return out


def merge_consecutive_speakers(segments, max_gap=1.5):
    """같은 화자의 연속 세그먼트 병합"""
    if not segments:
        return []
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        last = merged[-1]
        if seg["speaker"] == last["speaker"] and seg["start"] - last["end"] <= max_gap:
            last["end"] = seg["end"]
            last["text"] += " " + seg["text"]
        else:
            merged.append(seg.copy())
    return merged


def format_timestamp(s):
    return f"{int(s // 3600):02d}:{int((s % 3600) // 60):02d}:{int(s % 60):02d}"


def load_pipeline(token):
    """pyannote 3.2/3.3 인자명 호환"""
    try:
        return Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
    except TypeError:
        return Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)


def to_annotation(result):
    """pyannote 3.4+ 의 DiarizeOutput → Annotation 추출 (구버전은 그대로 반환)"""
    return getattr(result, "speaker_diarization", result)


def main():
    parser = argparse.ArgumentParser(description="YouTube 전사 + 화자 분리")
    parser.add_argument("url", help="YouTube URL")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--keep-audio", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if not HF_TOKEN:
        sys.exit("❌ HF_TOKEN 미설정 (.env 파일 또는 환경변수)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = os.path.join(OUTPUT_DIR, f"audio_{datetime.now():%Y%m%d_%H%M%S}")

    print("🎬 YouTube 다운로드...")
    audio_file, title = download_audio(args.url, base_name)
    print(f"   {title}")

    print(f"\n🚀 Whisper 전사 (language={args.language}, threads={args.threads})...")
    whisper_result, whisper_json = run_whisper(audio_file, args.language, args.threads)
    segments = whisper_result.get("transcription", [])
    print(f"   세그먼트 {len(segments)}개")
    if not segments:
        sys.exit("❌ 전사 결과 없음")

    print("\n🎙️ pyannote 화자 분리...")
    device_str = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"   장치: {device_str}")

    pipeline = load_pipeline(HF_TOKEN)
    if pipeline is None:
        sys.exit(
            "❌ pyannote 파이프라인 로드 실패. HF_TOKEN 유효성과 모델 라이선스 동의 확인:\n"
            "   - https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "   - https://huggingface.co/pyannote/segmentation-3.0"
        )
    pipeline.to(torch.device(device_str))

    audio_dict = load_audio_for_pyannote(audio_file)
    diar_kwargs = {}
    if args.num_speakers is not None:
        diar_kwargs["num_speakers"] = args.num_speakers
    else:
        if args.min_speakers is not None:
            diar_kwargs["min_speakers"] = args.min_speakers
        if args.max_speakers is not None:
            diar_kwargs["max_speakers"] = args.max_speakers

    diarization = to_annotation(pipeline(audio_dict, **diar_kwargs))
    speakers = sorted(set(spk for _, _, spk in diarization.itertracks(yield_label=True)))
    print(f"   감지 화자: {len(speakers)}명 {speakers}")

    final = merge_consecutive_speakers(assign_speakers(segments, diarization))

    output_txt = base_name + "_transcript.txt"
    output_json = base_name + "_transcript.json"

    print(f"\n✅ 완료 ({len(final)}개 발화)\n" + "-" * 80)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n# URL: {args.url}\n# 화자: {len(speakers)}명\n\n")
        for seg in final:
            line = f"[{format_timestamp(seg['start'])}] {seg['speaker']}: {seg['text']}"
            print(line)
            f.write(line + "\n")
    print("-" * 80)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"title": title, "url": args.url, "speakers": speakers,
                   "segments": final}, f, ensure_ascii=False, indent=2)

    print(f"\n📁 {output_txt}\n📁 {output_json}")

    if not args.keep_audio:
        for p in (audio_file, whisper_json):
            try:
                os.remove(p)
            except OSError:
                pass


if __name__ == "__main__":
    main()