import os
import cv2
import torch
import yaml
import numpy as np
from ultralytics import YOLO
from PIL import Image

class AccidentTypeClassifier:
    def __init__(self, config_path='config.yml'):
        # 설정 로드
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # YOLO 모델 로드
        self.model = self._load_model()
        
        # 결과 저장 디렉토리 생성
        os.makedirs(self.config['video']['output_path'], exist_ok=True)
    
    def _load_model(self):
        # YOLO 모델 로드 및 설정
        model = YOLO(self.config['model']['path'])
        model.to(self.device)
        return model
    
    def classify_accident(self, frame, detections):
        """사고 유형 분류 로직"""
        accident_type = None
        confidence = 0
        
        if len(detections) == 0:
            return None, 0
        
        # 검출된 객체들의 위치와 크기를 기반으로 사고 유형 분류
        boxes = detections[0].boxes
        if len(boxes) >= 2:  # 최소 2개 이상의 객체가 검출되어야 함
            # 바운딩 박스들의 IoU(Intersection over Union) 계산
            ious = self._calculate_ious(boxes.xyxy)
            
            # IoU와 객체 클래스를 기반으로 사고 유형 분류
            accident_type, confidence = self._determine_accident_type(
                boxes.cls,
                boxes.conf,
                ious
            )
        
        return accident_type, confidence
    
    def _calculate_ious(self, boxes):
        """박스들 간의 IoU 계산"""
        ious = torch.zeros((len(boxes), len(boxes)))
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                ious[i, j] = self._box_iou(boxes[i], boxes[j])
        return ious
    
    def _box_iou(self, box1, box2):
        """두 박스 간의 IoU 계산"""
        x1 = torch.max(box1[0], box2[0])
        y1 = torch.max(box1[1], box2[1])
        x2 = torch.min(box1[2], box2[2])
        y2 = torch.min(box1[3], box2[3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / (box1_area + box2_area - intersection + 1e-6)
    
    def _determine_accident_type(self, classes, confidences, ious):
        """검출된 객체들의 관계를 분석하여 사고 유형 결정"""
        # 설정된 사고 유형 매핑
        accident_types = self.config['model']['accident_types']
        
        max_iou = torch.max(ious).item()
        classes = classes.cpu().numpy()
        confidences = confidences.cpu().numpy()
        
        # 높은 IoU를 가진 객체쌍 찾기
        high_iou_mask = ious > self.config['detection']['iou_threshold']
        if not torch.any(high_iou_mask):
            return None, 0
        
        # 객체 클래스 조합에 따른 사고 유형 분류
        for i, j in zip(*torch.where(high_iou_mask)):
            class1, class2 = int(classes[i]), int(classes[j])
            conf1, conf2 = confidences[i], confidences[j]
            
            # 차량-차량 충돌
            if class1 == class2 == accident_types['vehicle']:
                return 'vehicle_collision', min(conf1, conf2)
            
            # 차량-보행자 사고
            elif {class1, class2} == {accident_types['vehicle'], accident_types['pedestrian']}:
                return 'pedestrian_accident', min(conf1, conf2)
            
            # 차량-자전거 사고
            elif {class1, class2} == {accident_types['vehicle'], accident_types['bicycle']}:
                return 'bicycle_accident', min(conf1, conf2)
        
        return None, 0
    
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
            
            # YOLO 객체 검출
            results = self.model(frame)
            
            # 사고 유형 분류
            accident_type, confidence = self.classify_accident(frame, results)
            
            # 사고 감지된 경우 저장
            if accident_type and confidence > self.config['detection']['confidence_threshold']:
                accident_frames.append({
                    'frame': frame,
                    'timestamp': frame_count / cap.get(cv2.CAP_PROP_FPS),
                    'type': accident_type,
                    'confidence': confidence
                })
                
                # 사고 프레임 저장
                if self.config['logging']['save_frames']:
                    # 검출 결과 시각화
                    annotated_frame = results[0].plot()
                    cv2.putText(
                        annotated_frame,
                        f"Type: {accident_type} ({confidence:.2f})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    
                    cv2.imwrite(
                        os.path.join(
                            self.config['video']['output_path'],
                            f'accident_frame_{frame_count}_{accident_type}.jpg'
                        ),
                        annotated_frame
                    )
                
                # 로깅
                if self.config['logging']['verbose']:
                    print(f"Accident detected: {accident_type} at {frame_count/cap.get(cv2.CAP_PROP_FPS):.2f} seconds")
        
        cap.release()
        return accident_frames
    
    def generate_report(self, accident_frames):
        # 사고 감지 결과 리포트 생성
        report_path = os.path.join(self.config['video']['output_path'], 'detection_report.txt')
        with open(report_path, 'w') as f:
            f.write("Traffic Accident Classification Report\n")
            f.write("===================================\n\n")
            f.write(f"Video: {self.config['video']['input_path']}\n")
            f.write(f"Total Accidents Detected: {len(accident_frames)}\n\n")
            
            # 사고 유형별 통계
            accident_types = {}
            for frame_info in accident_frames:
                accident_type = frame_info['type']
                accident_types[accident_type] = accident_types.get(accident_type, 0) + 1
            
            f.write("Accident Type Statistics:\n")
            for accident_type, count in accident_types.items():
                f.write(f"  {accident_type}: {count}\n")
            f.write("\n")
            
            # 개별 사고 정보
            for idx, frame_info in enumerate(accident_frames, 1):
                f.write(f"Accident {idx}:\n")
                f.write(f"  Type: {frame_info['type']}\n")
                f.write(f"  Timestamp: {frame_info['timestamp']:.2f} seconds\n")
                f.write(f"  Confidence: {frame_info['confidence']:.2f}\n")
                f.write("\n")
        
        print(f"Detailed report saved to {report_path}")

def main():
    classifier = AccidentTypeClassifier('config.yml')
    accident_frames = classifier.detect_accidents_in_video()
    classifier.generate_report(accident_frames)

if __name__ == '__main__':
    main()