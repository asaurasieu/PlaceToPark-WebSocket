import os
import subprocess
import signal
import sys

output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

youtube_url = "https://www.youtube.com/live/EPKWu223XEg?si=js_LWhj38TicLhQY"


command = (
    f'yt-dlp -f best -o - "{youtube_url}" '
    f'| ffmpeg -loglevel verbose -i pipe:0 -vf "fps=1/7" {output_dir}/frame_%04d.jpg'
)

process = None

def quit_process(signum, frame):
    print("Exiting...")
    if process and process.poll() is None:
        process.terminate()
    sys.exit(0)


signal.signal(signal.SIGINT, quit_process)

if __name__ == "__main__":
    try:
        print("[INFO] Fetching live stream via yt-dlp and piping to FFmpeg...")
        process = subprocess.Popen(command, shell=True)
        process.wait()  
    except KeyboardInterrupt:
        quit_process(None, None)
