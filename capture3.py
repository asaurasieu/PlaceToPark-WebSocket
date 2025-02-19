import os
import subprocess
import signal

# Directory to save frames
output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

# Replace this with your actual YouTube live stream URL
youtube_url = "https://www.youtube.com/live/EPKWu223XEg?si=js_LWhj38TicLhQY"

# Use yt-dlp to get the proper stream URL
yt_dlp_command = f'yt-dlp -g "{youtube_url}"'
stream_url = subprocess.check_output(yt_dlp_command, shell=True).decode().strip()

# FFmpeg command to capture frames
ffmpeg_command = f'ffmpeg -loglevel verbose -i "{stream_url}" -vf "fps=1/5" {output_dir}/frame_%04d.jpg'

def capture_frames():
    """Start FFmpeg to capture frames."""
    process = subprocess.Popen(ffmpeg_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process

def quit_process(signum, frame):
    """Graceful exit on Ctrl+C."""
    print("Exiting...")
    if process:
        process.terminate()
    exit(0)

# Signal handler for Ctrl+C
signal.signal(signal.SIGINT, quit_process)

if __name__ == "__main__":
    try:
        print("Fetching live stream and capturing frames...")
        process = capture_frames()
        process.wait()
    except KeyboardInterrupt:
        quit_process(None, None)
