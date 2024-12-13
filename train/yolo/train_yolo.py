import os
from ultralytics import YOLO
import yaml

class TrafficAccidentYOLOTrainer:
    def __init__(self, config_path):
        # 설정 파일 로드
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 클래스 설정
        self.classes = {
            "0": "Car-to-Bicycle-Crash", 
            "1": "Car-to-Car-Crash", 
            "2": "Car-to-Motorcycle-Crash", 
            "3": "Car-to-Pedestrian-Crash"
        }
        
        # YOLO 모델 초기화 (사전 훈련된 모델 또는 새 모델)
        self.model = YOLO(self.config['pretrained_model_path'])
    
    def prepare_dataset_yaml(self):
        # YOLO 학습을 위한 데이터셋 YAML 생성
        dataset_config = {
            'path': self.config['data_path'],
            'train': 'train/images',
            'val': 'valid/images',
            'nc': len(self.classes),
            'names': list(self.classes.values())
        }
        
        # YAML 파일로 저장
        with open(os.path.join(self.config['data_path'], 'data.yaml'), 'w') as f:
            yaml.dump(dataset_config, f)
    
    def train(self):
        # 데이터셋 YAML 준비
        self.prepare_dataset_yaml()
        
        # YOLO 모델 학습
        results = self.model.train(
            data=os.path.join(self.config['data_path'], 'data.yaml'),
            epochs=self.config['epochs'],
            batch=self.config['batch_size'],
            imgsz=self.config['image_size'],
            lr=self.config['learning_rate']
        )
        
        # 최종 모델 저장
        best_model_path = os.path.join(
            self.config['model_save_path'], 
            'best_yolo_model.pt'
        )
        self.model.save(best_model_path)
        print(f"Best model saved to {best_model_path}")

# 사용 예시
if __name__ == '__main__':
    trainer = TrafficAccidentYOLOTrainer('train_yolo.yml')
    trainer.train()