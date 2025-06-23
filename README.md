# MEC-Based-Real-Time-Object-Detection-using-YOLOv5-

Multi-access Edge Computing (MEC) is a transformative technology in 5G networks that allows data processing to occur close to the data source, reducing latency and enhancing efficiency. One practical application of MEC is real-time object detection, where video feeds from devices (UEs) are analyzed on a nearby MEC server rather than in the cloud or the device itself.

This documentation outlines a setup where a webcam feed from one machine (Machine-1, the Transmitter) is transmitted to another machine (Machine-2, the MEC Server), which performs object detection using YOLOv5. The detections are then visualized through a simple web dashboard.

This project demonstrates a simple real-time object detection pipeline using YOLOv5, where:

- Machine-1 (Transmitter) captures webcam video and streams it via RTSP.

- Machine-2 (MEC Server) receives the RTSP stream, processes it using YOLOv5, and hosts a web dashboard to display detections.

- Detected object data is stored in InfluxDB for visualization and analytics.

## 🖥️ System Architectur
Meachine 1 (webcam) ------> MEC server(Object Detect) ----> HTTP(Web Dashboar) -----> Infinixdb(Database)

## Transmitter Side (Machine-1)

#### Identify Available Camera Devices

```bash
ls /dev/video*
```

#### Install and Start RTSP Streaming
installation 
```bash
sudo apt install v4l2rtspserver
sudo snap connect sudo  v4l2rtspserver:camera
sudo v4l2rtspserver /dev/video0
```
Make sure the device has proper camera permissions.
    • The default RTSP URL will be: rtsp://<machine1_ip>:8554/unicast
Note: Both machines must be on the same local network.

another option use the code provided to start the rtsp server 

## MEC Server Side (Machine-2)
Requiremented Packages: 
 - flask
 - opencv-python-headless
 - torch
 - torchvision
 - numpy
 - matplotlib
 - pillow
 - scipy
 - pandas

#### Install with
```bash
pip install flask torch torchvision opencv-python numpy
```

#### Python Code Summary
 - Flask server to handle video uploads and stream processing.
 - YOLOv5s model from ultralytics via torch.hub.
 - stream_processor() function processes RTSP stream.
 - Captured frames with detections are stored in a detections/ folder.
 - dashboard route displays the latest 20 detections as a webpage.



RTSP Capture in Python:
```python
“cap = cv2.VideoCapture("rtsp://192.168.0.116:8554/unicast")”
```
Change ip address that you got in machine one 

This captures the live stream from Machine-1 and processes it frame-by-frame.

#### Detections Folder:

- All output images with object detection annotations are saved here with timestamps.

#### Web Interface Access:
- After running the Python script, the web dashboard is accessible at
           link came on terminal.
on terminal we will be getting like this

Features:
- Displays latest 20 detection images.
- Auto-refreshes every 10 seconds.
- Simple and lightweight interface for edge analytics visualization.


#### Uploading into Influxdb :
  in code change this fields according to your influxdb 
```python
- influx_url = ""      ex: http://192.168.0.109:8086
- influx_token = ""    ex: YBSoXq_NdBU8Wj8USP4c_NjAng0cD1yDYSQSK0iInP2A9ZPmJbnmI64tDBjUMouGJH7q321Xfujxmk47zpRmvw==
- influx_org = ""      ex: crl
- influx_bucket = ""   ex: objectdetection
```
Note : Dont use data provided in example. 
  
