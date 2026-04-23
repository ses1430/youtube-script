import yt_dlp
from faster_whisper import WhisperModel
import os
import sys
import time
import hashlib

if len(sys.argv) < 2:
    print("사용법: python youtube-script.py [URL]")
    sys.exit(1)

url = sys.argv[1]

# URL 기반 고유 파일명 생성 (동시 실행 충돌 방지)
url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
audio_file = f"temp_audio_{url_hash}.mp3"
ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': f'temp_audio_{url_hash}.%(ext)s',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
    }],
    'quiet': True,
    'remote_components': 'ejs:github',
}

print("오디오 다운로드 중...")
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

print("Whisper로 전사 시작... (CPU 모드)")
transcribe_start = time.time()

# AMD GPU라서 CPU로 실행 (가장 안정적)
model = WhisperModel(
    #"large-v3",           # 정확도 최고
    "turbo",              # 속도 최적화 (대부분 영상에서 충분히 정확)
    device="cpu",         # ← AMD라서 cpu로 고정
    compute_type="int8"   # 메모리 적게 쓰고 속도 빠름
)

segments, info = model.transcribe(
    audio_file,
    beam_size=5,
    language=None,  # 자동 감지
    vad_filter=True
)

print("\n=== 전사 결과 ===\n")
full_text = ""
for segment in segments:
    print(f"[{segment.start:.2f}s → {segment.end:.2f}s] {segment.text}")
    full_text += segment.text + " "

print("\n=== 전체 스크립트 ===\n")
print(full_text)

elapsed = time.time() - transcribe_start
print(f"\n전사 소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")

output_file = f"transcript_{url_hash}.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(full_text.strip())
print(f"\n파일 저장 완료: {output_file}")

os.remove(audio_file)
