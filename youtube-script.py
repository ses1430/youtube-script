import argparse
import yt_dlp
import subprocess
import os
import json
import warnings
from datetime import datetime

# ==================== Warning 억제 ====================
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio")
os.environ["PYANNOTE_AUDIO_NO_TORCHCODEC"] = "1"
# ====================================================

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("url", nargs="?", help="YouTube URL")
    group.add_argument("-f", "--file", help="로컬 오디오 파일 경로")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--threads", type=int, default=6)

    args = parser.parse_args()

    # ==================== 경로 설정 ====================
    WHISPER_DIR = r"D:\PYTHON\youtube-script\whisper.cpp-windows-vulkan"
    WHISPER_EXE = os.path.join(WHISPER_DIR, "whisper-cli.exe")
    MODEL_PATH = os.path.join(WHISPER_DIR, "ggml-large-v3-turbo-q5_0.bin")
    #MODEL_PATH = os.path.join(WHISPER_DIR, "ggml-large-v3-q5_0.bin")
    OUTPUT_DIR = r"D:\PYTHON\youtube-script\res"
    # ==================================================

    if args.file:
        # ── 파일 모드 ──────────────────────────────────
        audio_file = os.path.abspath(args.file)
        if not os.path.exists(audio_file):
            print(f"❌ 파일을 찾을 수 없습니다: {audio_file}")
            return
        stem = os.path.splitext(audio_file)[0]
        txt_file = stem + ".txt"
        cleanup = []
    else:
        # ── URL 모드 ───────────────────────────────────
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.join(OUTPUT_DIR, f"audio_{timestamp}")
        audio_file = base_name + ".mp3"
        txt_file = base_name + ".txt"

        print(f"🎬 YouTube 영상 다운로드 중...")
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': base_name,
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([args.url])

        if not os.path.exists(audio_file):
            print(f"❌ 오디오 파일을 찾을 수 없습니다.")
            return
        cleanup = [audio_file]

    print(f"\n🚀 Whisper 전사 시작... (language={args.language}, threads={args.threads})")

    cmd = [
        WHISPER_EXE, "-m", MODEL_PATH, "-f", audio_file,
        "--language", args.language,
        "--threads", str(args.threads),
        "--output-json",
        "--temperature", "0",
        "--best-of", "5",
        "--no-speech-thold", "0.8",
    ]
    subprocess.run(cmd, text=True)

    whisper_json = audio_file + ".json"
    if not os.path.exists(whisper_json):
        print("❌ Whisper JSON 파일을 찾을 수 없습니다.")
        return
    cleanup.append(whisper_json)

    with open(whisper_json, 'r', encoding='utf-8', errors='replace') as f:
        whisper_result = json.load(f)

    # ==================== 반복 제거 + 텍스트 저장 ====================
    segments = whisper_result.get("transcription", [])
    cleaned_segments = []
    prev_text = ""

    for seg in segments:
        text = seg["text"].strip()
        if text == prev_text:
            continue
        cleaned_segments.append(text)
        prev_text = text

    with open(txt_file, "w", encoding="utf-8") as f:
        for text in cleaned_segments:
            f.write(text + "\n")

    print(f"\n✅ 전사 완료!")
    print(f"📁 결과 파일: {txt_file}")


if __name__ == "__main__":
    main()
