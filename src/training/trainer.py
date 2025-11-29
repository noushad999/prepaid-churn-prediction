import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional
import mlflow
import logging
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json

logger = logging.getLogger(__name__)
class ChurnPredictionTrainer:
    """
    Production-grade trainer with MLflow tracking, early stopping, and checkpointing.
    """
    
    def __init__(self, model: nn.Module, config: Dict, device: str = 'cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Training parameters
        self.epochs = config['training']['epochs']
        self.batch_size = config['training']['batch_size']
        self.learning_rate = config['training']['learning_rate']
        self.weight_decay = config['training']['weight_decay']
        self.early_stopping_patience = config['training']['early_stopping_patience']
        
        # Optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Loss function (weighted for imbalanced data)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
        # MLflow setup
        mlflow.set_experiment(config['mlflow']['experiment_name'])
        
    def _create_optimizer(self):
        optimizer_name = self.config['training']['optimizer']
        if optimizer_name == 'AdamW':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_name == 'Adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_scheduler(self):
        scheduler_name = self.config['training']['scheduler']
        if scheduler_name == 'CosineAnnealingLR':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs
            )
        elif scheduler_name == 'ReduceLROnPlateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=5
            )
        else:
            return None
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict]:
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        metrics = self._calculate_metrics(all_labels, all_preds)
        
        return avg_loss, metrics
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        metrics = self._calculate_metrics(all_labels, all_preds, all_probs)
        
        return avg_loss, metrics
    
    def _calculate_metrics(self, labels, preds, probs=None) -> Dict:
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, average='binary', zero_division=0),
            'recall': recall_score(labels, preds, average='binary', zero_division=0),
            'f1': f1_score(labels, preds, average='binary', zero_division=0)
        }
        
        if probs is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(labels, probs)
            except:
                metrics['auc_roc'] = 0.0
        
        return metrics
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            checkpoint_dir: str = 'checkpoints/'):
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        with mlflow.start_run():
            # Log config
            mlflow.log_params(self.config['training'])
            mlflow.log_params(self.config['model']['ccp_net'])
            
            for epoch in range(self.epochs):
                # Train
                train_loss, train_metrics = self.train_epoch(train_loader)
                
                # Validate
                val_loss, val_metrics = self.validate(val_loader)
                
                # Update scheduler
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # Log metrics
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    **{f'train_{k}': v for k, v in train_metrics.items()},
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                }, step=epoch)
                
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.4f}, "
                    f"Val F1: {val_metrics['f1']:.4f}"
                )
                
                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.best_model_state = self.model.state_dict().copy()
                    
                    # Save checkpoint
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_metrics': val_metrics
                    }, f"{checkpoint_dir}/best_model.pth")
                    
                    mlflow.log_artifact(f"{checkpoint_dir}/best_model.pth")
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
            # Load best model
            self.model.load_state_dict(self.best_model_state)
            
            return self.model
