# main_simple.py - Simplified without complex imports
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import ADASYN
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("  TELECOM CHURN PREDICTION - SIMPLIFIED VERSION")
print("="*70)

# Create directories
Path('checkpoints').mkdir(exist_ok=True)
Path('logs').mkdir(exist_ok=True)

# 1. Load data
print("\n1. Loading data...")
df = pd.read_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"   ✓ Loaded {len(df):,} records")

# 2. Preprocess
print("\n2. Preprocessing...")
df = df.drop(columns=['customerID'], errors='ignore')
df = df.dropna()

# Encode categorical features
for col in df.select_dtypes(include=['object']).columns:
    if col != 'Churn':
        df[col] = LabelEncoder().fit_transform(df[col])

# Separate features and target
X = df.drop(columns=['Churn']).values
y = LabelEncoder().fit_transform(df['Churn'])

# Scale features
X = StandardScaler().fit_transform(X)

# Balance dataset
print(f"   Original: {np.bincount(y)}")
X, y = ADASYN(random_state=42).fit_resample(X, y)
print(f"   Balanced: {np.bincount(y)}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   ✓ Train: {len(X_train):,}, Test: {len(X_test):,}")

# 3. Build model
print("\n3. Building deep neural network...")

class ChurnPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        return self.network(x)

model = ChurnPredictor(X_train.shape[1])
print(f"   ✓ Model with {sum(p.numel() for p in model.parameters()):,} parameters")

# 4. Training setup
print("\n4. Training setup...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
print(f"   ✓ Using device: {device}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

# Create data loaders
train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
    batch_size=64, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
    batch_size=64, shuffle=False
)

# 5. Training loop
print("\n5. Training model...")
print("-" * 70)

best_acc = 0
patience = 0
max_patience = 10
epochs = 50

for epoch in range(epochs):
    # Train
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    
    # Validate
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    avg_val_loss = val_loss / len(test_loader)
    val_acc = accuracy_score(all_labels, all_preds)
    
    # Learning rate scheduling
    scheduler.step(avg_val_loss)
    
    # Print progress
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    # Early stopping
    if val_acc > best_acc:
        best_acc = val_acc
        patience = 0
        torch.save(model.state_dict(), 'checkpoints/best_model.pth')
    else:
        patience += 1
        if patience >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

# 6. Final evaluation
print("\n" + "="*70)
print("  FINAL EVALUATION")
print("="*70)

model.load_state_dict(torch.load('checkpoints/best_model.pth'))
model.eval()

with torch.no_grad():
    X_test_t = torch.FloatTensor(X_test).to(device)
    outputs = model(X_test_t)
    y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

print("\n" + classification_report(y_test, y_pred, 
                                   target_names=['No Churn', 'Churn'],
                                   digits=4))

print("="*70)
print(f"  ✓ Best Accuracy: {best_acc:.2%}")
print(f"  ✓ Model saved: checkpoints/best_model.pth")
print("="*70)
