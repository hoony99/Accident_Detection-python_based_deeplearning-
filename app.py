import os
import atexit
import time
import threading
from contextlib import contextmanager
from flask import Flask, request, render_template, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import densenet121
import torch.nn as nn
from PIL import Image
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
from ultralytics import YOLO
import plyer

# Flask 앱 초기화
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)
app.config['SECRET_KEY'] = os.urandom(24).hex()
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'mov', 'mp4', 'avi'}
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
# 파일 정리 관리자
class FileCleanupManager:
    def __init__(self):
        self.cleanup_list = set()
        self._lock = threading.Lock()

    def add_file(self, filepath):
        with self._lock:
            self.cleanup_list.add(filepath)

    def cleanup(self):
        with self._lock:
            for filepath in list(self.cleanup_list):
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        self.cleanup_list.remove(filepath)
                except Exception:
                    pass

    def schedule_cleanup(self, filepath):
        self.add_file(filepath)
        threading.Timer(60.0, self.try_cleanup, args=(filepath,)).start()

    def try_cleanup(self, filepath):
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                with self._lock:
                    self.cleanup_list.discard(filepath)
        except Exception:
            pass

cleanup_manager = FileCleanupManager()
atexit.register(cleanup_manager.cleanup)

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
            app_icon=None,
            timeout=10
        )
    except Exception as e:
        app.logger.error(f"Failed to send notification: {e}")

# 필요한 디렉토리 생성
def create_directories():
    directories = [
        app.config['UPLOAD_FOLDER'],
        app.static_folder,
        os.path.join(app.static_folder, 'accidents'),
        os.path.join(BASE_DIR, 'logs'),
        os.path.join(BASE_DIR, 'templates')  # templates 디렉토리도 확인
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        app.logger.info(f"Directory created/verified: {directory}")
# 모델 로드 함수들
def load_densenet_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = densenet121(weights=None)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, 2)
        
        model_path = os.path.join(BASE_DIR, 'models', 'densenet_model50.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"DenseNet model not found at {model_path}")
            
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        app.logger.error(f"Error loading DenseNet model: {e}")
        raise

def load_yolo_model():
    try:
        model_path = os.path.join(BASE_DIR, 'models', 'YOLOv8.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found at {model_path}")
        return YOLO(model_path)
    except Exception as e:
        app.logger.error(f"Error loading YOLO model: {e}")
        raise

# 파일 확장자 검사
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# 비디오 캡처 컨텍스트 매니저
@contextmanager
def safe_video_capture(video_path):
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        yield cap
    finally:
        if cap is not None:
            cap.release()

# 비디오 처리 함수
def process_video(video_path, densenet_model, yolo_model, device):
    try:
        with safe_video_capture(video_path) as cap:
            if not cap.isOpened():
                raise Exception("Failed to open video file")

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            frame_count = 0
            accident_count = 0
            last_accident_frame = None
            last_accident_info = None
            frames_to_skip = 1  # 기본 스킵 프레임 수

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                # 스킵할 프레임 수만큼 건너뛰기
                if frame_count % frames_to_skip != 0:
                    continue

                # OpenCV BGR을 RGB로 변환하고 PIL Image로 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # 이미지 전처리
                input_tensor = transform(pil_image)
                input_tensor = input_tensor.unsqueeze(0).to(device)

                # DenseNet 예측
                with torch.no_grad():
                    output = densenet_model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    confidence = probabilities[0][0].item()  # 사고 클래스(0)의 확률
                    predicted = torch.argmax(probabilities, dim=1)

                    # 사고 확률에 따른 다음 프레임 스킵 수 설정
                    if confidence >= 0.5:
                        frames_to_skip = 1  # 바로 다음 프레임 검사
                    elif confidence >= 0.4:
                        frames_to_skip = 2  # 2프레임 후 검사
                    elif confidence >= 0.3:
                        frames_to_skip = 3  # 3프레임 후 검사
                    elif confidence >= 0.2:
                        frames_to_skip = 4  # 4프레임 후 검사
                    else:
                        frames_to_skip = 5  # 5프레임 후 검사

                    if predicted.item() == 0:  # 사고 감지
                        accident_count += 1
                        # 마지막 사고 프레임과 정보 저장
                        last_accident_frame = frame.copy()
                        
                        # YOLO 분석
                        results_yolo = yolo_model(frame)
                        accident_type = "차량 사고"
                        
                        for result in results_yolo:
                            boxes = result.boxes
                            if len(boxes) > 0:
                                cls = boxes[0].cls.cpu().numpy()[0]
                                if str(int(cls)) in yolo_model.model.names:
                                    accident_type = yolo_model.model.names[str(int(cls))]

                        last_accident_info = {
                            "type": accident_type,
                            "confidence": f"{confidence * 100:.2f}%",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # 로깅 추가
                        app.logger.info(f"Accident detected at frame {frame_count} with confidence {confidence:.2f}")
                        app.logger.info(f"Next frame skip: {frames_to_skip}")
                    else:
                        accident_count = 0

                    # 연속 3프레임 이상 사고가 감지되면 마지막 프레임 저장 및 반환
                    if accident_count >= 3 and last_accident_frame is not None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_name = f'accident_{timestamp}.jpg'
                        save_path = os.path.join(app.static_folder, 'accidents', save_name)
                        
                        # 이미지 저장
                        cv2.imwrite(save_path, last_accident_frame)
                        
                        last_accident_info['frame_path'] = f'accidents/{save_name}'
                        
                        # 알림 전송
                        send_notification(
                            "교통사고 감지",
                            f"사고 유형: {last_accident_info['type']}\n정확도: {last_accident_info['confidence']}"
                        )
                        
                        return [last_accident_info]  # 마지막 프레임 정보만 반환

            return None

    except Exception as e:
        app.logger.error(f"Error in process_video: {e}")
        raise

# 라우트: 메인 페이지
@app.route('/')
def index():
    return render_template('index.html')

# 라우트: 비디오 업로드 및 처리
@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            app.logger.info("No video file in request")
            flash('비디오 파일을 선택해주세요')
            return redirect(url_for('index'))

        file = request.files['video']
        if file.filename == '':
            app.logger.info("No selected filename")
            flash('파일이 선택되지 않았습니다')
            return redirect(url_for('index'))

        if not allowed_file(file.filename):
            app.logger.info(f"Invalid file type: {file.filename}")
            flash('지원하지 않는 파일 형식입니다')
            return redirect(url_for('index'))

        try:
            # 파일 저장
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)
            app.logger.info(f"File saved to {video_path}")

            # 모델 로드
            app.logger.info("Loading models...")
            densenet_model, device = load_densenet_model()
            yolo_model = load_yolo_model()
            app.logger.info("Models loaded successfully")

            # 비디오 처리
            app.logger.info("Processing video...")
            accidents = process_video(video_path, densenet_model, yolo_model, device)
            app.logger.info(f"Video processed. Found {len(accidents) if accidents else 0} accidents")

            # 파일 정리 예약
            cleanup_manager.schedule_cleanup(video_path)

            # 결과 렌더링 시도
            try:
                app.logger.info("Attempting to render results template...")
                app.logger.info(f"Accidents data: {accidents}")
                return render_template('results.html', accidents=accidents)
            except Exception as template_error:
                app.logger.error(f"Template rendering error: {template_error}", exc_info=True)
                flash('결과 페이지 로딩 중 오류가 발생했습니다')
                return redirect(url_for('index'))

        except Exception as processing_error:
            app.logger.error(f"Video processing error: {processing_error}", exc_info=True)
            if 'video_path' in locals() and os.path.exists(video_path):
                cleanup_manager.schedule_cleanup(video_path)
            flash('비디오 처리 중 오류가 발생했습니다')
            return redirect(url_for('index'))

    except Exception as e:
        app.logger.error(f"Upload error: {e}", exc_info=True)
        flash('파일 업로드 중 오류가 발생했습니다')
        return redirect(url_for('index'))

# 정적 파일 제공
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    create_directories()
    setup_logging()
    
    # 이전 업로드 파일 정리
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cleanup_manager.schedule_cleanup(file_path)
    
    app.run(debug=True, port=5000)