import argparse
import yt_dlp
import subprocess
import os
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="YouTube Whisper Transcription")
    parser.add_argument("url", help="YouTube URL")
    parser.add_argument("--language", default="auto", help="언어 코드 (기본: auto)")
    parser.add_argument("--threads", type=int, default=6, help="CPU 스레드 수 (기본: 6)")

    args = parser.parse_args()

    # ==================== 경로 설정 ====================
    WHISPER_DIR = r"D:\PYTHON\youtube-script\whisper.cpp-windows-vulkan"
    WHISPER_EXE = os.path.join(WHISPER_DIR, "whisper-cli.exe")
    MODEL_PATH = os.path.join(WHISPER_DIR, "ggml-large-v3-turbo-q5_0.bin")
    OUTPUT_DIR = r"D:\PYTHON\youtube-script\whisper-output"
    # ==================================================

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ★ base_name으로 깔끔하게 관리
    base_name = os.path.join(OUTPUT_DIR, f"audio_{timestamp}")
    audio_file = base_name + ".mp3"      # audio_20260426_000230.mp3
    txt_file = base_name + ".txt"        # audio_20260426_000230.txt   ← 원하는 형태

    print(f"🎬 YouTube 영상 다운로드 중...")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': base_name,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }],
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([args.url])

    if not os.path.exists(audio_file):
        print(f"❌ 오디오 파일을 찾을 수 없습니다: {audio_file}")
        return

    print(f"🚀 Whisper 전사 시작... (language={args.language}, threads={args.threads})")

    cmd = [
        WHISPER_EXE,
        "-m", MODEL_PATH,
        "-f", audio_file,
        "--language", args.language,
        "--threads", str(args.threads),
        "--output-txt"
    ]

    result = subprocess.run(cmd, text=True)

    if result.returncode == 0:
        # whisper가 생성한 audio_xxx.mp3.txt → audio_xxx.txt 로 변경
        wrong_txt = audio_file + ".txt"
        if os.path.exists(wrong_txt):
            os.rename(wrong_txt, txt_file)

        print(f"\n✅ 전사 완료!")
        print(f"📁 결과 파일: {txt_file}")
    else:
        print("❌ 에러 발생:")
        print(result.stderr)


if __name__ == "__main__":
    main()