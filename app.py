import os
from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import densenet121
import torch.nn as nn
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
from ultralytics import YOLO
import boto3
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
import plyer  # 데스크톱 알림용

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24).hex()
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mov', 'mp4', 'avi'}

# 환경변수 로드
load_dotenv()

# AWS S3 설정
S3_IMG_BUCKET = os.getenv('S3_IMG_BUCKET')
S3_VIDEO_BUCKET = os.getenv('S3_VIDEO_BUCKET')
S3_KEY = os.getenv('S3_KEY')
S3_SECRET = os.getenv('S3_SECRET')
S3_REGION = os.getenv('S3_REGION')

# S3 클라이언트 설정
s3_client = boto3.client(
    's3',
    aws_access_key_id=S3_KEY,
    aws_secret_access_key=S3_SECRET,
    region_name=S3_REGION
)

# 로깅 설정
def setup_logging():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logging.basicConfig(level=logging.INFO)
    handler = RotatingFileHandler('logs/app.log', maxBytes=10000, backupCount=3)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)

# 알림 전송 함수
def send_notification(title, message):
    try:
        plyer.notification.notify(
            title=title,
            message=message,
            app_icon=None,  # 아이콘 경로 설정 가능
            timeout=10  # 알림 표시 시간(초)
        )
    except Exception as e:
        app.logger.error(f"Failed to send notification: {e}")

# DenseNet 모델 로드
def load_densenet_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load('densenet_model50.pth', map_location=device))
    model.to(device)
    model.eval()
    return model, device

# YOLO 모델 로드
def load_yolo_model(model_path):
    return YOLO(model_path)

# 파일 확장자 검사
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# 비디오 처리 함수
def process_video(video_path, densenet_model, yolo_model, device):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        app.logger.error("Failed to open video source")
        return None

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    frame_count = 0
    accident_count = 0
    frame_skip = 1
    accidents_detected = []

    while cap.isOpened() and accident_count < 3:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            input_tensor = transform(frame).unsqueeze(0).to(device)

            with torch.no_grad():
                output = densenet_model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

                if predicted.item() == 1:  # 사고 감지
                    accident_count += 1
                    if accident_count >= 3:
                        # YOLO로 사고 유형 분석
                        results_yolo = yolo_model(frame)
                        accident_type = "unknown"
                        for result in results_yolo:
                            boxes = result.boxes
                            for box in boxes:
                                cls = int(box.cls[0].item())
                                if str(cls) in yolo_model.model.names:
                                    accident_type = yolo_model.model.names[str(cls)]
                                    break

                        # 사고 프레임 저장
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_path = os.path.join(app.config['UPLOAD_FOLDER'], f'accident_{timestamp}.jpg')
                        cv2.imwrite(save_path, frame)

                        accident_info = {
                            "type": accident_type,
                            "confidence": f"{confidence.item() * 100:.2f}%",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "frame_path": save_path
                        }
                        accidents_detected.append(accident_info)

                        # 알림 전송
                        send_notification(
                            "교통사고 감지",
                            f"사고 유형: {accident_type}\n정확도: {confidence.item() * 100:.2f}%"
                        )

        frame_count += 1

    cap.release()
    return accidents_detected

# 라우트: 메인 페이지
@app.route('/')
def index():
    return render_template('index.html')

# 라우트: 비디오 업로드 및 처리
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        flash('No video file uploaded')
        return redirect(url_for('index'))

    file = request.files['video']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        try:
            # 파일 저장
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)

            # 모델 로드
            densenet_model, device = load_densenet_model()
            yolo_model = load_yolo_model('YOLOv8.pt')

            # 비디오 처리
            accidents = process_video(video_path, densenet_model, yolo_model, device)

            if accidents:
                return render_template(
                    'results.html',
                    accidents=accidents
                )
            else:
                flash('No accidents detected in the video')
                return redirect(url_for('index'))

        except Exception as e:
            app.logger.error(f"Error processing video: {e}")
            flash('Error processing video')
            return redirect(url_for('index'))

    flash('Invalid file type')
    return redirect(url_for('index'))

if __name__ == '__main__':
    # 필요한 디렉토리 생성
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # 로깅 설정
    setup_logging()
    
    # 서버 실행
    app.run(debug=True, port=5000)
if __name__ == '__main__':    
    app.run(host='0.0.0.0', port=5000, debug=True)