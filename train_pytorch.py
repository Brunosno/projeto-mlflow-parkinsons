import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f'Usando o dispositivo: {device}')

# 1. Carregar o Dataset
df = pd.read_csv("data/parkinsons_updrs.data")
drop_cols = ["subject#", "test_time", "motor_UPDRS", "total_UPDRS"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns]).values
y = df["motor_UPDRS"].values

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=SEED
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15, random_state=SEED
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=64, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=64, shuffle=False)

def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total = 0

    with torch.set_grad_enabled(is_train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            preds = model(xb)
            loss = criterion(preds, yb)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * xb.size(0)
            total += xb.size(0)

    mse = total_loss / total
    rmse = math.sqrt(mse)
    return mse, rmse

def plot_history(history, title="Histórico de treino"):
    epochs = range(1, len(history["train_rmse"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_rmse"], label="Train RMSE")
    plt.plot(epochs, history["val_rmse"], label="Val RMSE")
    plt.xlabel("Época")
    plt.ylabel("RMSE")
    plt.title(title + " - RMSE")
    plt.legend()
    filepath = f"{title.replace(' ', '_')}.png"
    plt.savefig(filepath)
    plt.close()
    print(f"Gráfico salvo como {filepath}")
    return filepath

def train_and_log_mlflow(model, run_name, train_loader, val_loader, test_loader, criterion, optimizer, epochs=150, params=None):
    history = {"train_rmse": [], "val_rmse": []}
    
    # Inicia a gravação do experimento no MLflow
    mlflow.set_experiment("Parkinsons_Motor_UPDRS_Optimization")
    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params(params)
            
        for epoch in range(epochs):
            tr_mse, tr_rmse = run_epoch(model, train_loader, criterion, optimizer)
            va_mse, va_rmse = run_epoch(model, val_loader, criterion)

            history["train_rmse"].append(tr_rmse)
            history["val_rmse"].append(va_rmse)

            # Rastrear as métricas ao longo do tempo a cada 20 épocas e na última
            if epoch % 20 == 0 or epoch == epochs - 1:
                mlflow.log_metric("train_rmse", tr_rmse, step=epoch)
                mlflow.log_metric("val_rmse", va_rmse, step=epoch)
                print(f"Epoch {epoch:03d} | train_rmse={tr_rmse:.4f} val_rmse={va_rmse:.4f}")
        
        # Registrar métricas finais
        mlflow.log_metric("train_rmse_final", history["train_rmse"][-1])
        mlflow.log_metric("val_rmse_final", history["val_rmse"][-1])
        
        # Avaliar no conjunto de TESTE e logar a métrica
        te_mse, te_rmse = run_epoch(model, test_loader, criterion)
        mlflow.log_metric("test_rmse_final", te_rmse)
        print(f"Resultado no Teste: test_rmse={te_rmse:.4f}")
        
        # Gerar o gráfico de curvas e salvá-lo como artefato no MLflow
        graph_path = plot_history(history, title=run_name)
        mlflow.log_artifact(graph_path)

        # Salvar o modelo PyTorch
        mlflow.pytorch.log_model(model, "model", input_example=X_train_t[:5].numpy())
        print(f"Run '{run_name}' salva no MLflow!")
        
    return history

# ---------------------------------------------------------
# EXPERIMENTO 1: Modelo Inicial (Sem Regularização)
# ---------------------------------------------------------
class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

print("\n--- Treinando Modelo Inicial ---")
model = MLPRegressor(X_train_t.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

params_inicial = {"model_type": "pytorch_mlp_base", "learning_rate": 1e-3, "epochs": 150}
history_inicial = train_and_log_mlflow(model, "Exp1_PyTorch_Base", train_loader, val_loader, test_loader, criterion, optimizer, epochs=150, params=params_inicial)


# ---------------------------------------------------------
# EXPERIMENTO 2: Modelo Regularizado (Dropout + Weight Decay)
# ---------------------------------------------------------
class MLPRegularizedRegressor(nn.Module):
    def __init__(self, input_dim, dropout=0.6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

print("\n--- Treinando Modelo Regularizado ---")
model_exp = MLPRegularizedRegressor(X_train_t.shape[1], dropout=0.6).to(device)
optimizer_exp = torch.optim.Adam(model_exp.parameters(), lr=1e-3, weight_decay=1e-2)

params_exp = {"model_type": "pytorch_mlp_regularized", "learning_rate": 1e-3, "weight_decay": 1e-2, "dropout": 0.6, "epochs": 150}
history_exp = train_and_log_mlflow(model_exp, "Exp2_PyTorch_Regularized", train_loader, val_loader, test_loader, criterion, optimizer_exp, epochs=150, params=params_exp)


# ---------------------------------------------------------
# EXPERIMENTO 3: Modelo Ótimo (Soft Dropout + Soft L2)
# ---------------------------------------------------------
class MLPOptimizedRegressor(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.net(x)

print("\n--- Treinando Modelo Ótimo ---")
model_opt = MLPOptimizedRegressor(X_train_t.shape[1], dropout=0.1).to(device)
# Reduzimos o weight decay de 1e-2 para 1e-5 (relaxamos a punição)
optimizer_opt = torch.optim.Adam(model_opt.parameters(), lr=1e-3, weight_decay=1e-5)

params_opt = {"model_type": "pytorch_mlp_optimized", "learning_rate": 1e-3, "weight_decay": 1e-5, "dropout": 0.1, "epochs": 150}
history_opt = train_and_log_mlflow(model_opt, "Exp3_PyTorch_Optimized", train_loader, val_loader, test_loader, criterion, optimizer_opt, epochs=150, params=params_opt)
