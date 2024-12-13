import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torch.optim.lr_scheduler import ReduceLROnPlateau

class TrafficAccidentTrainer:
    def __init__(self, config_path):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set up data transforms
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Prepare datasets and dataloaders
        self._prepare_data()
        
        # Initialize model
        self._init_model()
        
        # Initialize loss and optimizer
        self._init_training_components()
    
    def _prepare_data(self):
        # Load datasets
        dataset = datasets.ImageFolder(
            root=self.config['data_path'], 
            transform=self.data_transforms
        )
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False
        )
    
    def _init_model(self):
        # Load pre-existing model or create new
        if os.path.exists(self.config['pretrained_model_path']):
            self.model = models.densenet121(pretrained=False)
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_features, 2)
            self.model.load_state_dict(torch.load(self.config['pretrained_model_path']))
        else:
            self.model = models.densenet121(pretrained=True)
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_features, 2)
        
        self.model = self.model.to(self.device)
    
    def _init_training_components(self):
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config['learning_rate']
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            patience=3, 
            factor=0.1
        )
    
    def train(self):
        best_val_loss = float('inf')
        
        for epoch in range(self.config['epochs']):
            self.model.train()
            train_loss = 0.0
            
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            # Print metrics
            print(f'Epoch {epoch+1}/{self.config["epochs"]}')
            print(f'Train Loss: {train_loss/len(self.train_loader):.4f}')
            print(f'Val Loss: {val_loss/len(self.val_loader):.4f}')
            print(f'Val Accuracy: {100 * correct / total:.2f}%')
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 
                           os.path.join(self.config['model_save_path'], 'best_model.pth'))
        
        print("Training complete!")

# Usage
if __name__ == '__main__':
    trainer = TrafficAccidentTrainer('train_densenet.yml')
    trainer.train()