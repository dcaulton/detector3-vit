import os
import time
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
import base64 
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import sys
import datetime
import logging
import mlflow

print(f"[{datetime.datetime.now()}] >>> DETECTION1 CONTAINER STARTED <<<")
print(f"[{datetime.datetime.now()}] Python version: {sys.version}")
print(f"[{datetime.datetime.now()}] Attempting MQTT connection to mosquitto.mqtt.svc.cluster.local:1883...")
sys.stdout.flush()

load_dotenv()  # For local dev; in k8s use Secrets

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service.mlflow.svc.cluster.local:5000"))
mlflow.set_experiment("detection3")

processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")  # Or "facebook/detr-resnet-50" for classic; try "microsoft/conditional-detr-resnet-50" for faster convergence
model = AutoModelForObjectDetection.from_pretrained("SenseTime/deformable-detr").to('cuda')  # Deformable-DETR is a strong ViT-ish upgrade

# MQTT credentials (from Secrets in k8s)
MQTT_BROKER = os.getenv("MQTT_BROKER", "mqtt-broker.default.svc.cluster.local")  # adjust to your broker service
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_USER = os.getenv("MQTT_USER")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")
MQTT_TOPIC = "frigate/#"

def check_gpu():
    print(f"[{datetime.datetime.now()}] GPU TEST START")
    if torch.cuda.is_available():
        print(f"CUDA available! Device: {torch.cuda.get_device_name(0)}")
        # Simple GPU op: matrix multiply
        a = torch.randn(1000, 1000, device='cuda')
        b = torch.randn(1000, 1000, device='cuda')
        c = torch.matmul(a, b)
        print(f"GPU matmul result sample: {c[0][0].item():.2f}")
        torch.cuda.synchronize()  # Ensure completion
    else:
        print("No CUDA – falling back to CPU")
    print(f"[{datetime.datetime.now()}] GPU TEST END")
    sys.stdout.flush()

def process_image(image_bytes: bytes):
    start_time = time.perf_counter()

    cv_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    inputs = processor(images=cv_image, return_tensors="pt").to('cuda')

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process (COCO classes: id2label in model.config)
    target_sizes = torch.tensor([cv_image.shape[:2]])
    results = processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=target_sizes)[0]  # Tune threshold

    # Extract detections
    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score > 0.7:  # Focus on high-conf for nature (avoid noise)
            box = box.cpu().numpy().astype(int)
            class_name = model.config.id2label[label.item()]
            if 'animal' in class_name or class_name in ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']:  # Prioritize nature
                detections.append({'class': class_name, 'conf': float(score), 'box': box.tolist()})

    # Log to MLflow (same as detection2)
    mlflow.log_metric("num_detections", len(detections))
    mlflow.log_metric("top_confidence", max([d['conf'] for d in detections]) if detections else 0)
    # Annotate image with cv2.rectangle/text, save to artifact_path
    # Log artifact, detections.json, etc.

    # For zero-shot twist later: Switch to OWL-ViT and prompt texts=["deer", "bird in tree", "fox"]

    end_time = time.perf_counter()
    inference_time = ((end_time - start_time) * 1000)
    return inference_time, generated_path

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker")
        client.subscribe(MQTT_TOPIC)
        print(f"Subscribed to {MQTT_TOPIC}")
    else:
        print(f"Connection failed with code {rc}")
    check_gpu()

def on_message(client, userdata, msg):
    print(f"[{datetime.datetime.now()}] >>> RAW MQTT MESSAGE RECEIVED on topic {msg.topic} <<<")
    print(f"    Payload length: {len(msg.payload)} bytes, Payload type: {type(msg.payload)}, First 50 bytes (hex): {msg.payload[:50].hex()}")
    sys.stdout.flush()
    
    if not msg.topic.endswith('snapshot'):
        return

    with mlflow.start_run(run_name="detection3-aa"):
        mlflow.log_param("topic", msg.topic)
        
        # Handle payload: raw JPEG on snapshot topics
        image_bytes = msg.payload
        # If using frigate/events topic instead, it's base64: 
        # payload = json.loads(msg.payload)
        # image_bytes = base64.b64decode(payload["after"]["snapshot"])
        
        inference_time, artifact_path = process_image(image_bytes)
        
        # uncomment below, then run this to step through: kubectl exec -it detection2-bdf99b996-kj6cx -n detection2 -- python -m pdb /app/app.py
        # import pdb; pdb.set_trace()
        mlflow.log_metric("inference_time", inference_time)
        mlflow.log_artifact(artifact_path)
        
        mlflow.log_param("prompt", "miskatonic style")  # or dynamic
        # Optional: trigger downstream actions (e.g., publish result back to MQTT)

# Create and configure the client
client = mqtt.Client(client_id="detection3")
if MQTT_USER and MQTT_PASSWORD:
    client.username_pw_set(MQTT_USER, MQTT_PASSWORD)

client.on_connect = on_connect
client.on_message = on_message

# Connect and loop
client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
client.loop_forever()  # Blocks here; handles reconnects automatically
