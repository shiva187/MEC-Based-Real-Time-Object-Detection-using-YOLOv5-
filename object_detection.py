from flask import Flask, request, send_from_directory, render_template_string
import threading
import cv2
import torch
import os
import glob
from datetime import datetime
import numpy as np
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

app = Flask(__name__)

# ======== Load YOLOv5 model =========
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)

# ======== Create Output Directory =========
output_dir = "detections"
os.makedirs(output_dir, exist_ok=True)

# ======== InfluxDB Setup =========
influx_url = "http://192.168.0.109:8086"
influx_token = "YBSoXq_NdBU8Wj8USP4c_NjAng0cD1yDYWQSK0iInP2A9ZQmJbnmI64uDBjUMouGJH7q321Xfujxmk47zpRmvw=="
influx_org = "crl"
influx_bucket = "objectdetection"

client = InfluxDBClient(url=influx_url, token=influx_token, org=influx_org)
write_api = client.write_api(write_options=SYNCHRONOUS)

def log_person_count(count, source="stream"):
    try:
        point = Point("person_detection") \
            .tag("source", source) \
            .field("count", count) \
            .time(datetime.utcnow())
        write_api.write(bucket=influx_bucket, org=influx_org, record=point)
        print(f"[INFLUXDB] Successfully logged {count} persons from {source} at {datetime.utcnow()}")
    except Exception as e:
        print(f"[INFLUXDB ERROR] Failed to write data: {str(e)}")

# ======== Cleanup old images ========
def cleanup_old_images(max_images=1000):
    """Keep only the most recent images to prevent disk filling"""
    image_files = sorted(glob.glob(f"{output_dir}/*.jpg"), key=os.path.getmtime)
    if len(image_files) > max_images:
        for old_file in image_files[:-max_images]:
            try:
                os.remove(old_file)
                print(f"[CLEANUP] Removed old file: {old_file}")
            except Exception as e:
                print(f"[CLEANUP ERROR] Could not remove {old_file}: {str(e)}")

# ========= 🔁 Real-Time Stream Processing =========
def stream_processor():
    print("[INFO] Real-time stream processor started.")
    #cap = cv2.VideoCapture("rtsp://192.168.0.116:8554/unicast")
    cap = cv2.VideoCapture("http://192.168.0.130:8000/?action=stream")
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame)
        rendered = results.render()[0]

        # Count persons (class = 0 in COCO)
        person_count = sum(1 for *_, conf, cls in results.xyxy[0] if int(cls) == 0)
        if person_count > 0:
            log_person_count(person_count, source="stream")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{output_dir}/stream_{frame_id}_{ts}.jpg"
            cv2.imwrite(filename, rendered)
            print(f"[STREAM] Saved: {filename}")
            cleanup_old_images()  # Cleanup after saving new image

        frame_id += 1

    cap.release()

# ========= 📤 File Upload =========
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return "No file uploaded", 400

    file = request.files['video']
    npdata = np.frombuffer(file.read(), np.uint8)
    video_file = f"/tmp/upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    with open(video_file, 'wb') as f:
        f.write(npdata)

    print(f"[UPLOAD] Processing: {video_file}")
    cap = cv2.VideoCapture(video_file)
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        rendered = results.render()[0]

        # Count persons (class = 0 in COCO)
        person_count = sum(1 for *_, conf, cls in results.xyxy[0] if int(cls) == 0)
        if person_count > 0:
            log_person_count(person_count, source="upload")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{output_dir}/upload_{frame_id}_{ts}.jpg"
            cv2.imwrite(filename, rendered)
            print(f"[UPLOAD] Saved: {filename}")
            cleanup_old_images()

        frame_id += 1

    cap.release()
    os.remove(video_file)
    return "File processed", 200

# ========= 🌐 Serve Image Files =========
@app.route('/detections/<filename>')
def serve_image(filename):
    return send_from_directory(output_dir, filename)

# ========= 🖼️ Dashboard Page =========
@app.route('/dashboard')
def dashboard():
    # Get 100 most recent images
    image_files = sorted(glob.glob(f"{output_dir}/*.jpg"), key=os.path.getmtime, reverse=True)[:100]
    image_tags = [f'<div class="image-container"><img src="/detections/{os.path.basename(f)}"><p>{os.path.basename(f)}</p></div>' for f in image_files]

    html = f"""
    <html>
    <head>
        <title>MEC Detections Dashboard</title>
        <meta http-equiv="refresh" content="10">
        <style>
            body {{ font-family: sans-serif; text-align: center; }}
            .image-container {{ 
                display: inline-block;
                margin: 10px; 
                padding: 10px;
                background: #f5f5f5;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            img {{ 
                max-width: 400px;
                max-height: 300px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            p {{ 
                margin: 5px 0 0 0;
                font-size: 12px;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <h1>📸 Recent Detections (Last 100)</h1>
        <div class="gallery">
            {"".join(image_tags)}
        </div>
    </body>
    </html>
    """
    return render_template_string(html)

# ========= 🚀 Start Everything =========
if __name__ == '__main__':
    # Cleanup any existing old images on startup
    cleanup_old_images()
    
    # Start stream processor in background
    threading.Thread(target=stream_processor, daemon=True).start()
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
