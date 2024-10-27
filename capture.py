import os
import subprocess
import requests
import glob
import signal
import time 



# URL of the live stream from VLC
stream_url = "https://manifest.googlevideo.com/api/manifest/hls_variant/expire/1730084565/ei/daoeZ5LzDbycp-oP3YeHoQs/ip/81.35.177.6/id/EPKWu223XEg.4/source/yt_live_broadcast/requiressl/yes/xpc/EgVo2aDSNQ%3D%3D/hfr/1/playlist_duration/30/manifest_duration/30/maudio/1/spc/qtApAQLb8L1xZO-d3CPpPZtasFpneEgTagm8brWLvpmgPsWHalDwECpUdY7MURg/vprv/1/go/1/rqh/5/pacing/0/nvgoi/1/keepalive/yes/fexp/51312688%2C51326932/dover/11/itag/0/playlist_type/DVR/sparams/expire%2Cei%2Cip%2Cid%2Csource%2Crequiressl%2Cxpc%2Chfr%2Cplaylist_duration%2Cmanifest_duration%2Cmaudio%2Cspc%2Cvprv%2Cgo%2Crqh%2Citag%2Cplaylist_type/sig/AJfQdSswRQIgRmhmEn2l-JiuYcdn7OLxDDSFf7AZUhRdFJu-LZ5I37ACIQDnT09dyCRtiuuUlLCoYPmQ3Pf6zWRnXefjtylD7Fz5MA%3D%3D/file/index.m3u8"

output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

# URL of the Flask server to which we'll upload the frames
flask_server_url = "http://13.60.137.225:5000/upload"

ffmpeg_command = f'ffmpeg -i "{stream_url}" -vf "fps=1/5" {output_dir}/frame_%04d.jpg'


def capture_frames():
    process = subprocess.Popen(ffmpeg_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process

def upload_frame_to_server(frame_path):
    try:
        with open(frame_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(flask_server_url, files=files)
            if response.status_code == 200:
                print(f"Successfully uploaded {frame_path}")
            else:
                print(f"Failed to upload {frame_path}: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error uploading {frame_path}: {str(e)}")

def quit_process():
    print("Exiting...")
    if process:
        process.terminate()
    exit(0)

# Register the signal handler for quitting
signal.signal(signal.SIGINT, quit_process)


try: 
    process = capture_frames()
    
    while True: 
        frames = glob.glob(f"{output_dir}/*.jpg")
        for frame in frames: 
            upload_frame_to_server(frame)
            os.remove(frame)
        time.sleep(1)
except KeyboardInterrupt: 
    quit_process(None,None)
    
        
        
        

