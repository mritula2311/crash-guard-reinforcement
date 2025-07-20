
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- 2. Config ----------
DATA_PATH = "C:/piggy/feed.csv"  
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
HIDDEN_SIZE = 32
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------- 3. Load and preprocess ----------
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["created_at", "entry_id"])
X = df.drop(columns=["field3"])
y = df["field3"] - 1  # Convert to 0,1,2

X.fillna(0, inplace=True)
y.fillna(0, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE)

# ---------- 4. Model ----------
class CrashSeverityNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CrashSeverityNet(input_size=X.shape[1], hidden_size=HIDDEN_SIZE).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ---------- 5. Training ----------
def train_one_epoch(epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        pred = out.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)

    print(f"Epoch {epoch:3d} | Loss: {total_loss/total:.4f} | Accuracy: {correct/total:.4f}")

# ---------- 6. Evaluate ----------
def evaluate():
    model.eval()
    all_preds, all_true = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb).argmax(1).cpu()
            all_preds.append(preds)
            all_true.append(yb)

    all_preds = torch.cat(all_preds)
    all_true = torch.cat(all_true)

    print("\nFinal Test-set Evaluation:")
    print(classification_report(all_true, all_preds, target_names=["Low", "Medium", "High"]))

# ---------- 7. Main ----------
if __name__ == "__main__":
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(epoch)

    evaluate()

    # Optional: save model + scaler
    torch.save(model.state_dict(), "crash_severity_model.pt")
    import joblib
    joblib.dump(scaler, "scaler.joblib")
    print("Model and scaler saved ✓")
