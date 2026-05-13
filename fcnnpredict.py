#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pollutants-FCNN Framework: Contaminant Toxicity Prediction System Based on Real Tox21 Data
Author: Max Planck Institute for Computer and Biological Sciences Interdisciplinary Research Team
Data Source: Tox21 Challenge Dataset (https://bioinf.jku.at/research/DeepTox/)
Modified: 添加模型保存和GUI预测功能
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# GUI相关库
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLineEdit, QLabel, QTextEdit, QFileDialog,
                             QGroupBox, QScrollArea, QMessageBox, QSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

# ==================== 1. Data Download and Preprocessing Module ====================

class Tox21DataProcessor:
    """Tox21 Data Processor - Based on Real Data Sources"""
    
    def __init__(self):
        # URLs for real Tox21 data sources
        self.tox21_urls = {
            'train_dense': 'https://bioinf.jku.at/research/DeepTox/tox21_dense_train.csv.gz',
            'test_dense': 'https://bioinf.jku.at/research/DeepTox/tox21_dense_test.csv.gz',
            'train_labels': 'https://bioinf.jku.at/research/DeepTox/tox21_labels_train.csv.gz',
            'test_labels': 'https://bioinf.jku.at/research/DeepTox/tox21_labels_test.csv.gz'
        }
        
        # 12 toxicity endpoints
        self.toxicity_endpoints = [
            'NR-AhR', 'NR-AR', 'NR-AR-LBD', 'NR-ER', 'NR-ER-LBD', 
            'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 
            'SR-MMP', 'SR-p53', 'NR-Aromatase'
        ]
        
        # Alternative column names that might appear in the data
        self.alternative_endpoint_names = {
            'NR-AhR': ['NR.AhR', 'NR_AhR', 'AhR'],
            'NR-AR': ['NR.AR', 'NR_AR', 'AR'],
            'NR-AR-LBD': ['NR.AR.LBD', 'NR_AR_LBD', 'AR-LBD'],
            'NR-ER': ['NR.ER', 'NR_ER', 'ER'],
            'NR-ER-LBD': ['NR.ER.LBD', 'NR_ER_LBD', 'ER-LBD'],
            'NR-PPAR-gamma': ['NR.PPAR.gamma', 'NR_PPAR_gamma', 'PPAR'],
            'SR-ARE': ['SR.ARE', 'SR_ARE', 'ARE'],
            'SR-ATAD5': ['SR.ATAD5', 'SR_ATAD5', 'ATAD5'],
            'SR-HSE': ['SR.HSE', 'SR_HSE', 'HSE'],
            'SR-MMP': ['SR.MMP', 'SR_MMP', 'MMP'],
            'SR-p53': ['SR.p53', 'SR_p53', 'p53'],
            'NR-Aromatase': ['NR.Aromatase', 'NR_Aromatase', 'Aromatase']
        }
        
        # Toxicity type mapping (based on biological classification)
        self.toxicity_mapping = {
            'NR-AhR': 'genotoxicity',      # Genotoxicity
            'NR-AR': 'cell_toxicity',      # Cell toxicity
            'NR-AR-LBD': 'cell_toxicity',  # Cell toxicity
            'NR-ER': 'cell_toxicity',      # Cell toxicity
            'NR-ER-LBD': 'cell_toxicity',  # Cell toxicity
            'NR-PPAR-gamma': 'cell_toxicity', # Cell toxicity
            'SR-ARE': 'biological_toxicity', # Biological toxicity
            'SR-ATAD5': 'genotoxicity',    # Genotoxicity
            'SR-HSE': 'cell_toxicity',     # Cell toxicity
            'SR-MMP': 'cell_toxicity',     # Cell toxicity
            'SR-p53': 'genotoxicity',      # Genotoxicity
            'NR-Aromatase': 'neuro_toxicity' # Neurotoxicity
        }
    
    def download_tox21_data(self):
        """Download real Tox21 dataset"""
        print("🔄 Downloading Tox21 dataset...")
        
        # Create data directory
        os.makedirs('data/tox21', exist_ok=True)
        
        downloaded_files = {}
        for name, url in self.tox21_urls.items():
            try:
                print(f"📥 Downloading {name}...")
                response = requests.get(url, timeout=60, stream=True)
                if response.status_code == 200:
                    file_path = f'data/tox21/{name}.csv.gz'
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    downloaded_files[name] = file_path
                    print(f"✅ Successfully downloaded: {file_path}")
                else:
                    print(f"❌ Download failed for {name}: HTTP {response.status_code}")
            except Exception as e:
                print(f"❌ Download error for {name}: {str(e)}")
        
        return downloaded_files
    
    def find_toxicity_columns(self, df_columns):
        """Find toxicity endpoint columns in the dataframe"""
        found_columns = {}
        
        for endpoint in self.toxicity_endpoints:
            # Check for exact match
            if endpoint in df_columns:
                found_columns[endpoint] = endpoint
            else:
                # Check for alternative names
                for alt_name in self.alternative_endpoint_names.get(endpoint, []):
                    if alt_name in df_columns:
                        found_columns[endpoint] = alt_name
                        break
        
        return found_columns
    
    def load_and_preprocess_data(self, train_dense_file, test_dense_file, 
                                train_labels_file, test_labels_file):
        """Load and preprocess Tox21 data"""
        print("🔄 Loading and preprocessing data...")
        
        # Load feature data
        print("📊 Loading feature data...")
        train_features = pd.read_csv(train_dense_file, compression='gzip')
        test_features = pd.read_csv(test_dense_file, compression='gzip')
        
        print(f"   Training features shape: {train_features.shape}")
        print(f"   Test features shape: {test_features.shape}")
        
        # Drop non-numeric columns
        print("🔍 Checking for non-numeric columns...")
        train_features = train_features.select_dtypes(include=[np.number])
        test_features = test_features.select_dtypes(include=[np.number])
        
        # Load label data
        print("📋 Loading label data...")
        train_labels = pd.read_csv(train_labels_file, compression='gzip')
        test_labels = pd.read_csv(test_labels_file, compression='gzip')
        
        print(f"   Training labels shape: {train_labels.shape}")
        print(f"   Test labels shape: {test_labels.shape}")
        
        # Find toxicity endpoint columns
        print("🔍 Searching for toxicity endpoint columns...")
        train_tox_cols = self.find_toxicity_columns(train_labels.columns)
        test_tox_cols = self.find_toxicity_columns(test_labels.columns)
        
        # Use intersection of found columns
        common_endpoints = []
        for endpoint in self.toxicity_endpoints:
            if endpoint in train_tox_cols and endpoint in test_tox_cols:
                common_endpoints.append(endpoint)
        
        print(f"✅ Found {len(common_endpoints)} toxicity endpoints")
        
        # Extract the actual column names
        train_tox_actual_cols = [train_tox_cols[endpoint] for endpoint in common_endpoints]
        test_tox_actual_cols = [test_tox_cols[endpoint] for endpoint in common_endpoints]
        
        # Extract feature matrices
        X_train = train_features.values.astype(np.float32)
        X_test = test_features.values.astype(np.float32)
        
        # Extract label matrices
        y_train = train_labels[train_tox_actual_cols].values.astype(np.float32)
        y_test = test_labels[test_tox_actual_cols].values.astype(np.float32)
        
        print(f"   Feature dimensions: Train={X_train.shape}, Test={X_test.shape}")
        print(f"   Label dimensions: Train={y_train.shape}, Test={y_test.shape}")
        
        # Handle missing values
        print("🧹 Handling missing values...")
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)
        
        # For labels, replace NaN with -1 (will be masked during training)
        y_train = np.nan_to_num(y_train, nan=-1.0)
        y_test = np.nan_to_num(y_test, nan=-1.0)
        
        # Check label statistics
        print("📈 Label statistics:")
        total_samples = y_train.shape[0]
        total_labels = y_train.shape[0] * y_train.shape[1]
        valid_labels = 0
        
        for i, endpoint in enumerate(common_endpoints):
            valid_mask = y_train[:, i] != -1
            valid_count = np.sum(valid_mask)
            valid_labels += valid_count
            
            if valid_count > 0:
                pos_count = np.sum(y_train[valid_mask, i] == 1)
                pos_ratio = pos_count / valid_count
                total_valid_ratio = valid_count / total_samples * 100
                print(f"   {endpoint}: {pos_count}/{valid_count} positive ({pos_ratio:.2%}), "
                      f"{valid_count}/{total_samples} valid ({total_valid_ratio:.1f}%)")
        
        print(f"\n   Overall: {valid_labels}/{total_labels} valid labels ({valid_labels/total_labels*100:.1f}%)")
        print(f"   Average valid labels per sample: {valid_labels/total_samples:.2f}")
        
        # Feature standardization
        print("⚖️ Standardizing features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"✅ Data preprocessing completed:")
        print(f"   Training set: {X_train_scaled.shape[0]} samples, {X_train_scaled.shape[1]} features")
        print(f"   Test set: {X_test_scaled.shape[0]} samples, {X_test_scaled.shape[1]} features")
        print(f"   Toxicity endpoints: {len(common_endpoints)}")
        
        # Update endpoints list
        self.toxicity_endpoints = common_endpoints
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    def aggregate_toxicity_types(self, y_data):
        """Aggregate toxicity endpoints into 4 main toxicity types"""
        toxicity_types = ['biological_toxicity', 'cell_toxicity', 'neuro_toxicity', 'genotoxicity']
        aggregated = np.zeros((y_data.shape[0], len(toxicity_types)))
        
        for i, tox_type in enumerate(toxicity_types):
            # Find all endpoints corresponding to this toxicity type
            related_endpoints = []
            for j, endpoint in enumerate(self.toxicity_endpoints):
                base_endpoint = endpoint
                for orig_endpoint in self.toxicity_mapping.keys():
                    if orig_endpoint in endpoint or endpoint in orig_endpoint:
                        base_endpoint = orig_endpoint
                        break
                
                if base_endpoint in self.toxicity_mapping and self.toxicity_mapping[base_endpoint] == tox_type:
                    related_endpoints.append(j)
            
            if related_endpoints:
                for sample_idx in range(y_data.shape[0]):
                    valid_values = []
                    for endpoint_idx in related_endpoints:
                        if y_data[sample_idx, endpoint_idx] != -1:
                            valid_values.append(y_data[sample_idx, endpoint_idx])
                    
                    if valid_values:
                        aggregated[sample_idx, i] = np.max(valid_values)
                    else:
                        aggregated[sample_idx, i] = -1
        
        return aggregated

# ==================== 2. Neural Network Architecture Module ====================

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Calculate attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        output = self.w_o(attn_output)
        return output, attn_weights

class PollutantsFCNN(nn.Module):
    """Pollutants-FCNN Framework: Multi-task toxicity prediction network with attention mechanism"""
    
    def __init__(self, input_dim=801, hidden_dims=[1024, 512, 256], 
                 num_heads=8, num_tasks=4, dropout_rate=0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_tasks = num_tasks
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        # Input projection layer
        self.input_projection = nn.Linear(input_dim, hidden_dims[0])
        
        # FCNN layers
        self.fcnn_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        for i in range(len(hidden_dims)):
            if i == 0:
                self.fcnn_layers.append(nn.Linear(hidden_dims[0], hidden_dims[0]))
            else:
                self.fcnn_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.batch_norm_layers.append(nn.BatchNorm1d(hidden_dims[i]))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        # Attention mechanism
        self.attention = MultiHeadAttention(hidden_dims[-1], num_heads)
        self.attention_norm = nn.LayerNorm(hidden_dims[-1])
        
        # Task-specific layers
        self.task_layers = nn.ModuleList()
        for i in range(num_tasks):
            task_layer = nn.Sequential(
                nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dims[-1] // 2, 1)
            )
            self.task_layers.append(task_layer)
        
        # Weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        x = F.relu(x)
        
        # FCNN layers
        for i, (fc_layer, bn_layer, dropout) in enumerate(
            zip(self.fcnn_layers, self.batch_norm_layers, self.dropout_layers)):
            x = fc_layer(x)
            x = bn_layer(x)
            x = F.relu(x)
            x = dropout(x)
        
        # Add sequence dimension for attention mechanism
        x_seq = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Attention mechanism
        attn_output, attn_weights = self.attention(x_seq)
        attn_output = self.attention_norm(attn_output + x_seq)
        attn_output = attn_output.squeeze(1)  # [batch_size, hidden_dim]
        
        # Multi-task prediction
        outputs = []
        for task_layer in self.task_layers:
            task_output = task_layer(attn_output)
            outputs.append(task_output)
        
        return torch.cat(outputs, dim=1), attn_weights

# ==================== 3. Training and Evaluation Module ====================

class ToxicityPredictor:
    """Toxicity Predictor: Training, evaluation"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.toxicity_types = ['biological_toxicity', 'cell_toxicity', 'neuro_toxicity', 'genotoxicity']
        self.scaler = None  # 存储标准化器
    
    def save_model(self, model_path='models/toxicity_model.pth', scaler_path='models/scaler.pkl'):
        """保存模型和标准化器"""
        os.makedirs('models', exist_ok=True)
        
        # 保存模型
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.model.input_dim,
            'num_tasks': self.model.num_tasks,
            'hidden_dims': self.model.hidden_dims,
            'num_heads': self.model.num_heads,
            'dropout_rate': self.model.dropout_rate
        }
        torch.save(model_state, model_path)
        print(f"✅ 模型已保存至: {model_path}")
        
        # 保存标准化器
        if self.scaler is not None:
            import pickle
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"✅ 标准化器已保存至: {scaler_path}")
        
        return model_path
    
    @staticmethod
    def load_model(model_path='models/toxicity_model.pth', scaler_path='models/scaler.pkl', device='auto'):
        """加载模型和标准化器"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 加载模型
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=device)
        
        model = PollutantsFCNN(
            input_dim=checkpoint['input_dim'],
            hidden_dims=checkpoint['hidden_dims'],
            num_heads=checkpoint['num_heads'],
            num_tasks=checkpoint['num_tasks'],
            dropout_rate=checkpoint['dropout_rate']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # 加载标准化器
        scaler = None
        if os.path.exists(scaler_path):
            import pickle
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        
        predictor = ToxicityPredictor(model, device)
        predictor.scaler = scaler
        
        print(f"✅ 模型已从 {model_path} 加载")
        return predictor
    
    def train_model(self, X_train, y_train, X_val, y_val, 
                   epochs=100, batch_size=64, learning_rate=0.001):
        """Train the model with proper handling of missing labels"""
        print(f"🚀 Starting model training (device: {self.device})")
        
        # Convert data to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Get number of tasks from model or infer from data
        if hasattr(self.model, 'num_tasks'):
            num_tasks = self.model.num_tasks
        else:
            num_tasks = y_train.shape[1]
        
        # Calculate class weights for each task to handle imbalance
        pos_weights = []
        for i in range(num_tasks):
            valid_mask = y_train[:, i] != -1
            if np.sum(valid_mask) > 0:
                pos_count = np.sum(y_train[valid_mask, i] == 1)
                neg_count = np.sum(y_train[valid_mask, i] == 0)
                if pos_count > 0:
                    pos_weight = neg_count / pos_count
                    pos_weights.append(min(pos_weight, 10.0))  # Cap at 10 to avoid extreme weights
                else:
                    pos_weights.append(1.0)
            else:
                pos_weights.append(1.0)
        
        pos_weight_tensor = torch.tensor(pos_weights).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, reduction='none')
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        
        # Early stopping parameters
        early_stopping_patience = 20
        early_stopping_counter = 0
        
        # Training history
        train_losses = []
        val_losses = []
        val_aucs = []
        
        best_val_auc = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            total_valid_samples = 0
            
            # Shuffle training data
            indices = torch.randperm(len(X_train))
            
            # Batch training
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:min(i + batch_size, len(indices))]
                X_batch = X_train_tensor[batch_indices]
                y_batch = y_train_tensor[batch_indices]
                
                optimizer.zero_grad()
                outputs, _ = self.model(X_batch)
                
                # Create mask for valid labels (not -1)
                valid_mask = (y_batch != -1).float()
                
                # Calculate loss only on valid labels
                loss_per_sample = criterion(outputs, y_batch)
                masked_loss = (loss_per_sample * valid_mask).sum()
                
                # Count valid samples for averaging
                valid_count = valid_mask.sum()
                
                if valid_count > 0:
                    batch_loss = masked_loss / valid_count
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += masked_loss.item()
                    total_valid_samples += valid_count.item()
            
            if total_valid_samples > 0:
                train_loss /= total_valid_samples
            else:
                train_loss = 0
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_predictions = []
            val_targets = []
            val_valid_masks = []
            total_val_valid_samples = 0
            
            with torch.no_grad():
                for i in range(0, len(X_val), batch_size*2):
                    batch_end = min(i + batch_size*2, len(X_val))
                    X_batch = X_val_tensor[i:batch_end]
                    y_batch = y_val_tensor[i:batch_end]
                    
                    outputs, _ = self.model(X_batch)
                    
                    # Create mask for valid labels
                    valid_mask = (y_batch != -1).float()
                    
                    # Calculate validation loss
                    loss_per_sample = criterion(outputs, y_batch)
                    masked_loss = (loss_per_sample * valid_mask).sum()
                    valid_count = valid_mask.sum()
                    
                    if valid_count > 0:
                        batch_val_loss = masked_loss / valid_count
                        val_loss += masked_loss.item()
                        total_val_valid_samples += valid_count.item()
                    
                    predictions = torch.sigmoid(outputs).cpu().numpy()
                    val_predictions.extend(predictions)
                    val_targets.extend(y_batch.cpu().numpy())
                    val_valid_masks.extend(valid_mask.cpu().numpy())
            
            if total_val_valid_samples > 0:
                val_loss /= total_val_valid_samples
            else:
                val_loss = 0
            
            val_predictions = np.array(val_predictions)
            val_targets = np.array(val_targets)
            val_valid_masks = np.array(val_valid_masks)
            
            # Calculate validation AUC for each task
            task_aucs = []
            for i in range(num_tasks):
                # Only consider samples with valid labels
                valid_indices = val_valid_masks[:, i] == 1
                if np.sum(valid_indices) > 1:  # Need at least 2 samples
                    try:
                        # Check if we have both positive and negative samples
                        unique_classes = np.unique(val_targets[valid_indices, i])
                        if len(unique_classes) >= 2:
                            auc = roc_auc_score(val_targets[valid_indices, i], 
                                              val_predictions[valid_indices, i])
                            task_aucs.append(auc)
                        else:
                            # If only one class, AUC is 0.5
                            task_aucs.append(0.5)
                    except:
                        task_aucs.append(0.5)
                else:
                    task_aucs.append(0.5)
            
            val_auc = np.mean(task_aucs) if task_aucs else 0.5
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Record history
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_aucs.append(val_auc)
            
            # Save best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = self.model.state_dict().copy()
                early_stopping_counter = 0  # Reset counter
            else:
                early_stopping_counter += 1
            
            # Early stopping
            if early_stopping_counter >= early_stopping_patience:
                print(f"⚠️ Early stopping at epoch {epoch+1} (no improvement in validation AUC)")
                break
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val AUC: {val_auc:.4f}")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        print(f"✅ Training completed! Best validation AUC: {best_val_auc:.4f}")
        
        return train_losses, val_losses, val_aucs
    
    def evaluate_model(self, X_test, y_test, batch_size=64):
        """Evaluate model performance"""
        print("📊 Evaluating model performance...")
        
        self.model.eval()
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                batch_end = min(i + batch_size, len(X_test))
                X_batch = X_test_tensor[i:batch_end]
                
                outputs, _ = self.model(X_batch)
                pred = torch.sigmoid(outputs).cpu().numpy()
                
                predictions.extend(pred)
                targets.extend(y_test[i:batch_end])
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Get number of tasks
        if hasattr(self.model, 'num_tasks'):
            num_tasks = self.model.num_tasks
        else:
            num_tasks = predictions.shape[1] if len(predictions.shape) > 1 else 1
        
        # Calculate metrics
        results = {}
        toxicity_types_to_use = self.toxicity_types[:num_tasks]
        
        for i, tox_type in enumerate(toxicity_types_to_use):
            try:
                # Only consider valid targets (not -1)
                valid_indices = targets[:, i] != -1 if len(targets.shape) > 1 else targets != -1
                if np.sum(valid_indices) > 1:  # Need at least 2 samples
                    # Reshape for single task
                    if len(targets.shape) == 1:
                        target_vals = targets[valid_indices]
                        pred_vals = predictions[valid_indices]
                    else:
                        target_vals = targets[valid_indices, i]
                        pred_vals = predictions[valid_indices, i]
                    
                    # Check if we have both positive and negative samples
                    unique_classes = np.unique(target_vals)
                    if len(unique_classes) >= 2:
                        auc = roc_auc_score(target_vals, pred_vals)
                        
                        # Binary classification predictions
                        pred_binary = (pred_vals > 0.5).astype(int)
                        accuracy = accuracy_score(target_vals, pred_binary)
                        f1 = f1_score(target_vals, pred_binary)
                        
                        results[tox_type] = {
                            'AUC': auc,
                            'Accuracy': accuracy,
                            'F1-Score': f1,
                            'Valid_Samples': np.sum(valid_indices)
                        }
                    else:
                        results[tox_type] = {
                            'AUC': 0.5,
                            'Accuracy': 0.5,
                            'F1-Score': 0.0,
                            'Valid_Samples': np.sum(valid_indices)
                        }
                else:
                    results[tox_type] = {
                        'AUC': 0.5,
                        'Accuracy': 0.5,
                        'F1-Score': 0.0,
                        'Valid_Samples': np.sum(valid_indices)
                    }
            except Exception as e:
                print(f"⚠️ Error calculating metrics for {tox_type}: {e}")
                results[tox_type] = {
                    'AUC': 0.5,
                    'Accuracy': 0.5,
                    'F1-Score': 0.0,
                    'Valid_Samples': 0
                }
        
        # Macro averages
        valid_tasks = [tox for tox in toxicity_types_to_use 
                      if results[tox]['Valid_Samples'] > 0]
        
        if valid_tasks:
            macro_auc = np.mean([results[tox]['AUC'] for tox in valid_tasks])
            macro_acc = np.mean([results[tox]['Accuracy'] for tox in valid_tasks])
            macro_f1 = np.mean([results[tox]['F1-Score'] for tox in valid_tasks])
            
            results['Macro_Average'] = {
                'AUC': macro_auc,
                'Accuracy': macro_acc,
                'F1-Score': macro_f1,
                'Valid_Tasks': len(valid_tasks)
            }
        else:
            results['Macro_Average'] = {
                'AUC': 0.5,
                'Accuracy': 0.5,
                'F1-Score': 0.0,
                'Valid_Tasks': 0
            }
        
        # Print detailed results
        for tox_type in toxicity_types_to_use:
            print(f"\n{tox_type}:")
            print(f"  AUC: {results[tox_type]['AUC']:.4f}")
            print(f"  Accuracy: {results[tox_type]['Accuracy']:.4f}")
            print(f"  F1-Score: {results[tox_type]['F1-Score']:.4f}")
        
        print(f"\nMacro Averages:")
        print(f"  AUC: {results['Macro_Average']['AUC']:.4f}")
        print(f"  Accuracy: {results['Macro_Average']['Accuracy']:.4f}")
        print(f"  F1-Score: {results['Macro_Average']['F1-Score']:.4f}")
        
        return results, predictions, targets
    
    def predict_single_sample(self, features):
        """预测单个样本的毒性类型"""
        self.model.eval()
        
        # 转换为numpy数组
        if isinstance(features, list):
            features = np.array(features, dtype=np.float32)
        
        # 确保维度正确
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # 标准化
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features
        
        # 转换为tensor
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs, _ = self.model(features_tensor)
            predictions = torch.sigmoid(outputs).cpu().numpy()[0]
        
        # 整理结果
        results = {}
        for i, tox_type in enumerate(self.toxicity_types[:len(predictions)]):
            results[tox_type] = {
                'probability': float(predictions[i]),
                'prediction': 'Positive' if predictions[i] > 0.5 else 'Negative',
                'confidence': float(predictions[i]) if predictions[i] > 0.5 else float(1 - predictions[i])
            }
        
        return results

# ==================== 4. Visualization Module ====================

class Visualizer:
    """Result visualization"""
    
    @staticmethod
    def plot_training_history(train_losses, val_losses, val_aucs, filename='training_history.png'):
        """Plot training history"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].plot(train_losses, label='Training Loss')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(val_losses, label='Validation Loss', color='orange')
        axes[1].set_title('Validation Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(val_aucs, label='Validation AUC', color='green')
        axes[2].set_title('Validation AUC')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('AUC')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Saved training history plot to {filename}")
        plt.show()

# ==================== 5. GUI 模块 ====================

class TrainingThread(QThread):
    """训练线程"""
    log_signal = pyqtSignal(str)
    finish_signal = pyqtSignal(object, object)
    
    def __init__(self, processor, X_train, X_val, y_train, y_val):
        super().__init__()
        self.processor = processor
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
    
    def run(self):
        try:
            self.log_signal.emit("🧠 初始化模型...")
            # 创建模型
            model = PollutantsFCNN(input_dim=self.X_train.shape[1], num_tasks=self.y_train.shape[1])
            predictor = ToxicityPredictor(model)
            predictor.scaler = self.processor.scaler if hasattr(self.processor, 'scaler') else None
            
            self.log_signal.emit("🚀 开始训练模型...")
            # 训练模型
            train_losses, val_losses, val_aucs = predictor.train_model(
                self.X_train, self.y_train, self.X_val, self.y_val, 
                epochs=100, learning_rate=0.0005
            )
            
            self.log_signal.emit("💾 保存模型...")
            # 保存模型
            predictor.save_model()
            
            self.log_signal.emit("✅ 模型训练完成！")
            self.finish_signal.emit(predictor, (train_losses, val_losses, val_aucs))
            
        except Exception as e:
            self.log_signal.emit(f"❌ 训练出错: {str(e)}")

class ToxicityPredictionGUI(QMainWindow):
    """毒性预测GUI主窗口"""
    
    def __init__(self):
        super().__init__()
        self.predictor = None
        self.processor = Tox21DataProcessor()
        self.init_ui()
    
    def init_ui(self):
        # 窗口设置
        self.setWindowTitle("污染物毒性预测系统 - Pollutants-FCNN")
        self.setGeometry(100, 100, 1000, 700)
        
        # 主部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 1. 标题
        title_label = QLabel("Pollutants-FCNN 污染物毒性预测系统")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 2. 功能选择选项卡
        # 2.1 模型训练组
        train_group = QGroupBox("模型训练")
        train_layout = QVBoxLayout(train_group)
        
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setMaximumHeight(150)
        
        train_btn_layout = QHBoxLayout()
        self.download_data_btn = QPushButton("下载Tox21数据集")
        self.train_model_btn = QPushButton("训练模型")
        self.load_model_btn = QPushButton("加载已训练模型")
        
        self.download_data_btn.clicked.connect(self.download_data)
        self.train_model_btn.clicked.connect(self.start_training)
        self.load_model_btn.clicked.connect(self.load_model)
        
        train_btn_layout.addWidget(self.download_data_btn)
        train_btn_layout.addWidget(self.train_model_btn)
        train_btn_layout.addWidget(self.load_model_btn)
        
        train_layout.addLayout(train_btn_layout)
        train_layout.addWidget(self.train_log)
        
        # 2.2 预测组
        predict_group = QGroupBox("毒性预测")
        predict_layout = QVBoxLayout(predict_group)
        
        # 特征输入
        input_layout = QHBoxLayout()
        
        self.feature_count_spin = QSpinBox()
        self.feature_count_spin.setRange(1, 2000)
        self.feature_count_spin.setValue(801)  # 默认801维
        self.feature_count_spin.setPrefix("特征维度: ")
        
        self.load_features_btn = QPushButton("加载特征文件")
        self.clear_features_btn = QPushButton("清空输入")
        
        self.load_features_btn.clicked.connect(self.load_features_file)
        self.clear_features_btn.clicked.connect(self.clear_features)
        
        input_layout.addWidget(self.feature_count_spin)
        input_layout.addWidget(self.load_features_btn)
        input_layout.addWidget(self.clear_features_btn)
        
        # 特征输入区域
        self.feature_input = QTextEdit()
        self.feature_input.setPlaceholderText("请输入特征数据，每行一个特征值，共801个特征...")
        self.feature_input.setMaximumHeight(200)
        
        # 预测按钮
        self.predict_btn = QPushButton("开始预测")
        self.predict_btn.clicked.connect(self.predict_toxicity)
        self.predict_btn.setEnabled(False)
        
        # 预测结果
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setPlaceholderText("预测结果将显示在这里...")
        
        predict_layout.addLayout(input_layout)
        predict_layout.addWidget(QLabel("特征数据输入:"))
        predict_layout.addWidget(self.feature_input)
        predict_layout.addWidget(self.predict_btn)
        predict_layout.addWidget(QLabel("预测结果:"))
        predict_layout.addWidget(self.result_display)
        
        # 添加到主布局
        main_layout.addWidget(train_group)
        main_layout.addWidget(predict_group)
        
        # 状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("就绪 - 请先训练或加载模型")
    
    def log_message(self, msg):
        """添加日志信息"""
        self.train_log.append(msg)
        self.status_bar.showMessage(msg)
    
    def download_data(self):
        """下载数据集"""
        self.log_message("📥 开始下载Tox21数据集...")
        try:
            downloaded = self.processor.download_tox21_data()
            if downloaded:
                self.log_message("✅ 数据集下载完成！")
            else:
                self.log_message("❌ 数据集下载失败！")
        except Exception as e:
            self.log_message(f"❌ 下载出错: {str(e)}")
    
    def start_training(self):
        """开始训练模型"""
        try:
            # 检查数据文件
            data_files = {
                'train_dense': 'data/tox21/train_dense.csv.gz',
                'test_dense': 'data/tox21/test_dense.csv.gz',
                'train_labels': 'data/tox21/train_labels.csv.gz',
                'test_labels': 'data/tox21/test_labels.csv.gz'
            }
            
            if not all(os.path.exists(f) for f in data_files.values()):
                self.log_message("⚠️ 数据集不存在，正在下载...")
                downloaded = self.processor.download_tox21_data()
                if not downloaded:
                    self.log_message("❌ 数据集下载失败，无法继续训练！")
                    return
            
            # 加载和预处理数据
            self.log_message("🔄 加载和预处理数据...")
            X_train, X_test, y_train_raw, y_test_raw, scaler = self.processor.load_and_preprocess_data(
                data_files['train_dense'], data_files['test_dense'],
                data_files['train_labels'], data_files['test_labels']
            )
            
            self.processor.scaler = scaler
            
            # 聚合毒性类型
            self.log_message("🔗 聚合毒性类型...")
            y_train = self.processor.aggregate_toxicity_types(y_train_raw)
            y_test = self.processor.aggregate_toxicity_types(y_test_raw)
            
            # 分割验证集
            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            self.log_message(f"📊 数据准备完成 - 训练集: {X_train_final.shape[0]} 样本")
            
            # 创建训练线程
            self.training_thread = TrainingThread(
                self.processor, X_train_final, X_val, y_train_final, y_val
            )
            self.training_thread.log_signal.connect(self.log_message)
            self.training_thread.finish_signal.connect(self.training_finished)
            self.training_thread.start()
            
        except Exception as e:
            self.log_message(f"❌ 训练初始化出错: {str(e)}")
    
    def training_finished(self, predictor, history):
        """训练完成回调"""
        self.predictor = predictor
        self.predict_btn.setEnabled(True)
        self.log_message("✅ 模型训练完成，可进行预测！")
        
        # 绘制训练历史
        train_losses, val_losses, val_aucs = history
        Visualizer.plot_training_history(train_losses, val_losses, val_aucs)
    
    def load_model(self):
        """加载已训练模型"""
        model_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "models", "PyTorch模型 (*.pth *.pt)"
        )
        
        if model_path:
            try:
                self.log_message(f"📂 加载模型: {model_path}")
                self.predictor = ToxicityPredictor.load_model(model_path)
                self.predict_btn.setEnabled(True)
                self.log_message("✅ 模型加载成功！")
            except Exception as e:
                self.log_message(f"❌ 模型加载失败: {str(e)}")
    
    def load_features_file(self):
        """加载特征文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择特征文件", "", "文本文件 (*.txt *.csv);;所有文件 (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    features = f.read()
                self.feature_input.setText(features)
                self.log_message(f"✅ 已加载特征文件: {file_path}")
            except Exception as e:
                self.log_message(f"❌ 加载特征文件失败: {str(e)}")
    
    def clear_features(self):
        """清空特征输入"""
        self.feature_input.clear()
        self.result_display.clear()
    
    def predict_toxicity(self):
        """预测毒性"""
        if not self.predictor:
            QMessageBox.warning(self, "警告", "请先训练或加载模型！")
            return
        
        try:
            # 获取输入特征
            input_text = self.feature_input.toPlainText().strip()
            if not input_text:
                QMessageBox.warning(self, "警告", "请输入特征数据！")
                return
            
            # 解析特征
            features = []
            lines = input_text.split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    # 支持逗号/空格分隔
                    if ',' in line:
                        values = line.split(',')
                    elif ' ' in line:
                        values = line.split()
                    else:
                        values = [line]
                    
                    for val in values:
                        if val:
                            features.append(float(val))
            
            # 检查维度
            expected_dim = self.feature_count_spin.value()
            if len(features) != expected_dim:
                QMessageBox.warning(
                    self, "警告", 
                    f"特征维度不匹配！预期{expected_dim}维，实际{len(features)}维"
                )
                return
            
            self.log_message("🔮 开始预测...")
            
            # 预测
            results = self.predictor.predict_single_sample(features)
            
            # 显示结果
            result_text = "📋 毒性预测结果:\n"
            result_text += "=" * 50 + "\n"
            
            for tox_type, res in results.items():
                result_text += f"\n{tox_type}:\n"
                result_text += f"  概率值: {res['probability']:.4f}\n"
                result_text += f"  预测结果: {res['prediction']}\n"
                result_text += f"  置信度: {res['confidence']:.4f}\n"
            
            self.result_display.setText(result_text)
            self.log_message("✅ 预测完成！")
            
        except Exception as e:
            self.log_message(f"❌ 预测出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"预测失败: {str(e)}")

# ==================== 6. 主函数 ====================

def main():
    """主函数"""
    # 创建GUI应用
    app = QApplication(sys.argv)
    
    # 设置字体
    font = QFont("SimHei", 9)
    app.setFont(font)
    
    # 创建并显示主窗口
    gui = ToxicityPredictionGUI()
    gui.show()
    
    # 运行应用
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
