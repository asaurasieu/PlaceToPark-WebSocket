import os
import subprocess
import signal
import sys


os.makedirs(output_dir, exist_ok=True)

youtube_url = "https://www.youtube.com/watch?v=EPKWu223XEg"
command = (
    f'yt-dlp -f best -o - "{youtube_url}" '
    f'| ffmpeg -loglevel verbose -i pipe:0 -vf "fps=1/7" -pix_fmt rgb24 {output_dir}/frame_%04d.jpg'
)

def quit_process(signum, frame):
    print("Exiting...")
    if process and process.poll() is None:
        process.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, quit_process)


print("[INFO] Fetching live stream via yt-dlp and piping to FFmpeg...")
process = subprocess.Popen(command, shell=True)
process.wait()
