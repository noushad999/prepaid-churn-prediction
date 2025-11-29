# quick_start.py - Minimal version for quick testing
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import ADASYN

print("="*60)
print("QUICK START - Telecom Churn Prediction")
print("="*60)

# Load data
print("\n1. Loading data...")
df = pd.read_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"   ✓ Loaded {len(df)} records")

# Simple preprocessing
print("\n2. Preprocessing...")
df = df.drop(columns=['customerID'], errors='ignore')
df = df.dropna()

# Encode categorical
for col in df.select_dtypes(include=['object']).columns:
    if col != 'Churn':
        df[col] = LabelEncoder().fit_transform(df[col])

# Features and target
X = df.drop(columns=['Churn']).values
y = LabelEncoder().fit_transform(df['Churn'])

# Scale
X = StandardScaler().fit_transform(X)

# Balance with ADASYN
print(f"   Original distribution: {np.bincount(y)}")
X, y = ADASYN(random_state=42).fit_resample(X, y)
print(f"   Balanced distribution: {np.bincount(y)}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   ✓ Train: {len(X_train)}, Test: {len(X_test)}")

# Simple Neural Network
print("\n3. Building model...")
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        return self.network(x)

model = SimpleNN(X_train.shape[1])
print(f"   ✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# Training
print("\n4. Training...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Convert to tensors
X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.LongTensor(y_train).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.LongTensor(y_test).to(device)

# Train
epochs = 30
batch_size = 64

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for i in range(0, len(X_train_t), batch_size):
        batch_X = X_train_t[i:i+batch_size]
        batch_y = y_train_t[i:i+batch_size]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 5 == 0:
        print(f"   Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(X_train_t)*batch_size:.4f}")

# Evaluation
print("\n5. Evaluating...")
model.eval()
with torch.no_grad():
    outputs = model(X_test_t)
    _, predicted = torch.max(outputs, 1)
    
    y_pred = predicted.cpu().numpy()
    y_true = y_test_t.cpu().numpy()

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_true, y_pred, target_names=['No Churn', 'Churn']))

print("\n✓ Quick start completed!")
print("  Next: Run 'python main.py' for full advanced system")
