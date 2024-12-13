import os
import cv2
import torch
import yaml
import numpy as np
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F

class AccidentDetector:
    def __init__(self, config_path='test_config.yml'):
        # 설정 로드
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 로드
        self.model = self._load_model()
        
        # 전처리 변환
        self.transform = self._get_transform()
        
        # 결과 저장 디렉토리 생성
        os.makedirs(self.config['video']['output_path'], exist_ok=True)
    
    def _load_model(self):
        # Densenet121 모델 로드
        model = models.densenet121(weights=None)
        num_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_features, self.config['model']['num_classes'])
        
        # 사전 훈련된 가중치 로드
        model.load_state_dict(torch.load(self.config['model']['path']))
        model.to(self.device)
        model.eval()
        return model
    
    def _get_transform(self):
        return transforms.Compose([
            transforms.Resize(tuple(self.config['preprocessing']['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config['preprocessing']['mean'],
                std=self.config['preprocessing']['std']
            )
        ])
    
    def detect_accidents_in_video(self):
        # 비디오 읽기
        video_path = self.config['video']['input_path']
        cap = cv2.VideoCapture(video_path)
        
        # 결과 저장을 위한 변수
        frame_count = 0
        accident_frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 프레임 간격 체크
            frame_count += 1
            if frame_count % (cap.get(cv2.CAP_PROP_FPS) // self.config['video']['frame_interval']) != 0:
                continue
            
            # 프레임 전처리
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_tensor = self.transform(pil_frame).unsqueeze(0).to(self.device)
            
            # 모델 예측
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                class_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][class_idx].item()
            
            # 사고 감지 로직
            if class_idx == 1 and confidence > self.config['video']['confidence_threshold']:
                accident_frames.append({
                    'frame': frame,
                    'timestamp': frame_count / cap.get(cv2.CAP_PROP_FPS),
                    'confidence': confidence
                })
                
                # 사고 프레임 저장 옵션
                if self.config['logging']['save_frames']:
                    cv2.imwrite(
                        os.path.join(
                            self.config['video']['output_path'], 
                            f'accident_frame_{frame_count}.jpg'
                        ), 
                        frame
                    )
                
                # 로깅
                if self.config['logging']['verbose']:
                    print(f"Accident detected at {frame_count/cap.get(cv2.CAP_PROP_FPS):.2f} seconds")
        
        cap.release()
        return accident_frames
    
    def generate_report(self, accident_frames):
        # 사고 감지 결과 리포트 생성
        report_path = os.path.join(self.config['video']['output_path'], 'detection_report.txt')
        with open(report_path, 'w') as f:
            f.write("Accident Detection Report\n")
            f.write("=======================\n\n")
            f.write(f"Video: {self.config['video']['input_path']}\n")
            f.write(f"Total Accident Frames: {len(accident_frames)}\n\n")
            
            for idx, frame_info in enumerate(accident_frames, 1):
                f.write(f"Accident {idx}:\n")
                f.write(f"  Timestamp: {frame_info['timestamp']:.2f} seconds\n")
                f.write(f"  Confidence: {frame_info['confidence']:.2f}\n")
        
        print(f"Detailed report saved to {report_path}")

def main():
    detector = AccidentDetector('test_config.yml')
    accident_frames = detector.detect_accidents_in_video()
    detector.generate_report(accident_frames)

if __name__ == '__main__':
    main()