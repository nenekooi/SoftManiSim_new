# -*- coding: utf-8 -*-
# 导入必要的库
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error #, r2_score
import matplotlib.pyplot as plt
import random
import joblib
import os
import math # Transformer 需要 math 库

# --- 中文显示设置 ---
try:
    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("[信息] 尝试设置字体为 'SimHei' 以支持中文。")
except Exception as e:
    print(f"[警告] 设置 'SimHei' 字体失败: {e}")

# --- 1. 设置随机种子 (保持一致性) ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# --- 2. 设备配置 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# --- 3. 配置: 修改为路径 ---
DATA_FILE_PATH = 'D:/data/save_data_new/-x.xlsx' # 请确保这是你的数据路径
# 定义保存 Transformer 模型和 scalers 的目录
MODEL_SAVE_DIR = 'D:/data/save_model/transformer_residual_model' # 为 Transformer 模型创建新的保存目录
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- 4. 加载和预处理数据 ---
print(f"正在从以下路径加载数据: {DATA_FILE_PATH}")
if not os.path.exists(DATA_FILE_PATH):
    print(f"[错误] 数据文件未找到: {DATA_FILE_PATH}")
    exit()
try:
    df = pd.read_excel(DATA_FILE_PATH, engine='openpyxl')
    print(f"数据加载成功。形状: {df.shape}")
    required_cols = ['cblen1_mm', 'cblen2_mm', 'cblen3_mm',
                     'X_real_mm', 'Y_real_mm', 'Z_real_mm',
                     'sim_X_mm', 'sim_Y_mm', 'sim_Z_mm']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"[错误] 数据文件缺少必需的列: {missing}")
        exit()
except Exception as e:
    print(f"加载数据时出错: {e}")
    exit()

# --- 5. 计算残差 (作为目标) ---
print("正在计算残差...")
df['residual_X'] = df['X_real_mm'] - df['sim_X_mm']
df['residual_Y'] = df['Y_real_mm'] - df['sim_Y_mm']
df['residual_Z'] = df['Z_real_mm'] - df['sim_Z_mm']

initial_rows = len(df)
df = df.dropna(subset=required_cols + ['residual_X', 'residual_Y', 'residual_Z']).copy() # 确保残差列也没有 NaN
if len(df) < initial_rows:
    print(f"[警告] 原始数据或计算残差后包含 NaN，已删除 {initial_rows - len(df)} 行。剩余 {len(df)} 行。")
if len(df) == 0:
    print("[错误] 没有有效的训练数据。")
    exit()

# --- 6. 定义输入和目标 ---
input_features = df[['cblen1_mm', 'cblen2_mm', 'cblen3_mm']].values
target_residuals = df[['residual_X', 'residual_Y', 'residual_Z']].values

print(f"输入特征形状: {input_features.shape}")
print(f"目标残差形状: {target_residuals.shape}")

# --- 7. 划分训练集和测试集 ---
split_ratio = 0.1 # 10% 测试集
split_index = int(len(input_features) * (1 - split_ratio))

X_train_raw = input_features[:split_index]
X_test_raw = input_features[split_index:]
y_train_raw = target_residuals[:split_index]
y_test_raw = target_residuals[split_index:]

sim_test_raw = df[['sim_X_mm', 'sim_Y_mm', 'sim_Z_mm']].iloc[split_index:].values
real_test_raw = df[['X_real_mm', 'Y_real_mm', 'Z_real_mm']].iloc[split_index:].values

print(f"训练集大小: {len(X_train_raw)}")
print(f"测试集大小: {len(X_test_raw)}")

# --- 8. 数据归一化 (使用 MinMaxScaler) ---
input_scaler = MinMaxScaler()
residual_scaler = MinMaxScaler()

X_train = input_scaler.fit_transform(X_train_raw)
y_train = residual_scaler.fit_transform(y_train_raw)
X_test = input_scaler.transform(X_test_raw)
y_test = residual_scaler.transform(y_test_raw)

x_scaler_path = os.path.join(MODEL_SAVE_DIR, 'transformer_x_scaler.joblib')
y_scaler_path = os.path.join(MODEL_SAVE_DIR, 'transformer_residual_scaler.joblib')
joblib.dump(input_scaler, x_scaler_path)
joblib.dump(residual_scaler, y_scaler_path)
print(f"输入 Scaler 保存至: {x_scaler_path}")
print(f"残差 Scaler 保存至: {y_scaler_path}")

# --- 9. 创建序列数据 ---
seq_length = 30  # Transformer 也需要序列长度 (可调整)
print(f"创建长度为 {seq_length} 的序列...")

def create_sequences(inputs, targets, seq_length):
    in_seqs = []
    out_seqs = []
    if len(inputs) <= seq_length:
        print("[警告] 数据长度不足以创建序列。")
        return np.array([]), np.array([])
    for i in range(len(inputs) - seq_length):
        in_seqs.append(inputs[i:i+seq_length])
        out_seqs.append(targets[i+seq_length]) # 预测序列后下一个点的目标
    if not in_seqs:
        print("[警告] 未能创建任何序列。")
        return np.array([]), np.array([])
    return np.array(in_seqs), np.array(out_seqs)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)

if len(X_test_seq) > 0:
    sim_test_aligned = sim_test_raw[seq_length:]
    real_test_aligned = real_test_raw[seq_length:]
    min_len = min(len(y_test_seq), len(sim_test_aligned), len(real_test_aligned))
    y_test_seq = y_test_seq[:min_len]
    sim_test_aligned = sim_test_aligned[:min_len]
    real_test_aligned = real_test_aligned[:min_len]
    print(f"序列化后，用于评估/绘图的测试点数量: {min_len}")
else:
    print("[警告] 未能生成测试序列，无法进行评估和绘图。")
    sim_test_aligned = np.array([])
    real_test_aligned = np.array([])

# --- 10. 自定义 Dataset 和 DataLoader (PyTorch 格式) ---
class RobotDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

batch_size = 64

if len(X_train_seq) == 0:
    print("[错误] 无法创建训练 DataLoader，没有训练序列。")
    exit()

train_dataset = RobotDataset(X_train_seq, y_train_seq)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

if len(X_test_seq) > 0:
    test_dataset = RobotDataset(X_test_seq, y_test_seq)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
else:
    test_loader = None

# --- 11. 定义 Transformer 模型 ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: [max_len, 1, d_model] -> [seq_len, batch, d_model] for batch_first=False
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model] if batch_first=True for TransformerEncoderLayer
        # pe shape needs to be broadcastable to x.
        # Current pe: [max_len, 1, d_model]. We need [seq_len, batch_size, d_model] then permute if batch_first
        # Or, more simply, adapt pe to [1, max_len, d_model] for direct addition if x is [batch, seq_len, d_model]
        x = x + self.pe[:x.size(1), :].squeeze(1).unsqueeze(0) # Adjust pe slicing for batch_first input
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, dim_feedforward, output_size, dropout=0.1, seq_len=30):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_size, d_model) # Simple linear layer as embedding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_len + 1) # max_len should be >= seq_len
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # The output of TransformerEncoder is (seq_len, batch_size, d_model) if batch_first=False (default)
        # or (batch_size, seq_len, d_model) if batch_first=True.
        # We want to predict a single output based on the whole sequence.
        # We can take the output of the last token, or average over all tokens.
        # Here, let's take the output corresponding to the last input token of the sequence.
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, src):
        # src shape: (batch_size, seq_len, input_size)
        src = self.embedding(src) * math.sqrt(self.d_model) # Embed and scale
        src = self.pos_encoder(src) # Add positional encoding
        
        # TransformerEncoder expects (seq_len, batch_size, d_model) if batch_first=False
        # or (batch_size, seq_len, d_model) if batch_first=True. Our encoder_layer is batch_first=True.
        output = self.transformer_encoder(src) # Shape: (batch_size, seq_len, d_model)
        
        # Use the output of the last time step (token) for prediction
        output = output[:, -1, :] # Shape: (batch_size, d_model)
        
        output = self.fc_out(output) # Shape: (batch_size, output_size)
        return output

# --- 12. 初始化模型、损失函数、优化器 ---
input_dim = X_train_seq.shape[2]    # Should be 3 (cblen1, cblen2, cblen3)
d_model = 128                       # Embedding dimension and Transformer's internal dimension (feature size)
nhead = 4                           # Number of attention heads (must be a divisor of d_model)
num_encoder_layers = 3              # Number of Transformer encoder layers
dim_feedforward = 512               # Dimension of the feedforward network model in nn.TransformerEncoderLayer
output_dim = y_train_seq.shape[1]   # Should be 3 (residual_X, residual_Y, residual_Z)
dropout_rate = 0.1                  # Dropout rate

model = TransformerModel(input_size=input_dim,
                         d_model=d_model,
                         nhead=nhead,
                         num_encoder_layers=num_encoder_layers,
                         dim_feedforward=dim_feedforward,
                         output_size=output_dim,
                         dropout=dropout_rate,
                         seq_len=seq_length).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005) # Potentially smaller LR for Transformer
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)


print("\nTransformer 模型结构:")
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型总参数量: {total_params}")


# --- 13. 训练模型 ---
num_epochs = 100 # 可调整
train_losses = []
print("\n开始训练 Transformer 模型...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs_seq, targets_seq in train_loader:
        inputs_seq = inputs_seq.to(device)
        targets_seq = targets_seq.to(device)

        optimizer.zero_grad()
        outputs = model(inputs_seq)
        loss = criterion(outputs, targets_seq)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping can be helpful
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    # scheduler.step(avg_loss) # Step the scheduler

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')

print("训练完成。")

# --- 14. 绘制训练损失曲线 ---
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_losses, label='训练损失 (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Transformer 模型训练损失曲线')
plt.legend()
plt.grid(True, linestyle=':')
loss_plot_path = os.path.join(MODEL_SAVE_DIR, 'transformer_training_loss.png')
plt.savefig(loss_plot_path)
print(f"训练损失曲线图已保存至: {loss_plot_path}")
# plt.show()

# --- 15. 测试模型 ---
if test_loader is None:
    print("\n无测试数据，跳过评估。")
else:
    print("\n开始评估 Transformer 模型...")
    model.eval()
    all_preds_scaled = []
    all_targets_scaled = [] # To store the ground truth scaled residuals for fair comparison
    with torch.no_grad():
        for inputs_seq, targets_seq in test_loader:
            inputs_seq = inputs_seq.to(device)
            targets_seq_device = targets_seq.to(device) # Ground truth residuals (scaled) on device
            outputs = model(inputs_seq) # Predicted residuals (scaled)
            all_preds_scaled.append(outputs.cpu().numpy())
            all_targets_scaled.append(targets_seq_device.cpu().numpy()) # Store ground truth

    all_preds_scaled = np.concatenate(all_preds_scaled, axis=0)
    all_targets_scaled = np.concatenate(all_targets_scaled, axis=0)

    predicted_residuals_mm = residual_scaler.inverse_transform(all_preds_scaled)
    true_residuals_mm = residual_scaler.inverse_transform(all_targets_scaled) # Inverse transform ground truth

    print("\n--- 残差预测评估 (毫米) ---")
    mae_residual_x = mean_absolute_error(true_residuals_mm[:, 0], predicted_residuals_mm[:, 0])
    mae_residual_y = mean_absolute_error(true_residuals_mm[:, 1], predicted_residuals_mm[:, 1])
    mae_residual_z = mean_absolute_error(true_residuals_mm[:, 2], predicted_residuals_mm[:, 2])
    avg_mae_residual = np.mean([mae_residual_x, mae_residual_y, mae_residual_z])
    print(f"残差 MAE: X={mae_residual_x:.3f}, Y={mae_residual_y:.3f}, Z={mae_residual_z:.3f}")
    print(f"平均残差 MAE: {avg_mae_residual:.3f} mm")

    print("\n--- 最终轨迹评估 (毫米) ---")
    sim_test = sim_test_aligned
    real_test = real_test_aligned

    nan_in_sim = np.isnan(sim_test).any(axis=1)
    if nan_in_sim.any():
        print(f"[警告] {nan_in_sim.sum()} 个测试样本的原始仿真坐标包含 NaN，将在计算修正轨迹和 MAE 时跳过。")
        valid_indices = ~nan_in_sim
        sim_test = sim_test[valid_indices]
        real_test = real_test[valid_indices]
        predicted_residuals_mm = predicted_residuals_mm[valid_indices]
        # Ensure true_residuals_mm is also sliced if it's used for direct comparison here
        # true_residuals_mm = true_residuals_mm[valid_indices] # Already done if all_targets_scaled was based on y_test_seq which is aligned

    corrected_sim_mm = sim_test + predicted_residuals_mm

    mae_corrected_x = mean_absolute_error(real_test[:, 0], corrected_sim_mm[:, 0])
    mae_corrected_y = mean_absolute_error(real_test[:, 1], corrected_sim_mm[:, 1])
    mae_corrected_z = mean_absolute_error(real_test[:, 2], corrected_sim_mm[:, 2])
    mae_corrected_3d = np.mean(np.linalg.norm(real_test - corrected_sim_mm, axis=1))
    print(f"修正后轨迹 MAE: X={mae_corrected_x:.3f}, Y={mae_corrected_y:.3f}, Z={mae_corrected_z:.3f}")
    print(f"修正后轨迹 Overall 3D MAE: {mae_corrected_3d:.3f} mm")

    # --- 16. 绘制最终轨迹对比图 ---
    print("\n绘制 Transformer 最终轨迹对比图...")
    plot_indices = np.arange(len(real_test))

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    plot_title_final = f'真实轨迹 vs 修正后仿真轨迹 (Transformer, 3D MAE={mae_corrected_3d:.3f} mm)'

    axs[0].plot(plot_indices, real_test[:, 0], label='真实 X (mm)', color='darkorange', linewidth=1.5)
    axs[0].plot(plot_indices, corrected_sim_mm[:, 0], label=f'修正后仿真 X (MAE={mae_corrected_x:.2f}mm)', color='green', linestyle='--', linewidth=1) # Changed color
    axs[0].set_ylabel('X 坐标 (mm)')
    axs[0].set_title('X 轴坐标对比')
    axs[0].legend()
    axs[0].grid(True, linestyle=':')

    axs[1].plot(plot_indices, real_test[:, 1], label='真实 Y (mm)', color='darkorange', linewidth=1.5)
    axs[1].plot(plot_indices, corrected_sim_mm[:, 1], label=f'修正后仿真 Y (MAE={mae_corrected_y:.2f}mm)', color='green', linestyle='--', linewidth=1)
    axs[1].set_ylabel('Y 坐标 (mm)')
    axs[1].set_title('Y 轴坐标对比')
    axs[1].legend()
    axs[1].grid(True, linestyle=':')

    axs[2].plot(plot_indices, real_test[:, 2], label='真实 Z (mm)', color='darkorange', linewidth=1.5)
    axs[2].plot(plot_indices, corrected_sim_mm[:, 2], label=f'修正后仿真 Z (MAE={mae_corrected_z:.2f}mm)', color='green', linestyle='--', linewidth=1)
    axs[2].set_ylabel('Z 坐标 (mm)')
    axs[2].set_title('Z 轴坐标对比')
    axs[2].legend()
    axs[2].grid(True, linestyle=':')

    axs[2].set_xlabel('测试集样本序号 (Aligned Index)')
    fig.suptitle(plot_title_final, fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    comparison_plot_path = os.path.join(MODEL_SAVE_DIR, 'transformer_trajectory_comparison.png')
    try:
        plt.savefig(comparison_plot_path, dpi=300)
        print(f"最终轨迹对比图已保存至: {comparison_plot_path}")
    except Exception as e:
        print(f"保存对比图像时出错: {e}")
    plt.show()


# --- 17. 保存训练好的 PyTorch 模型 ---
model_save_path = os.path.join(MODEL_SAVE_DIR, 'transformer_residual_model.pth')
try:
    torch.save(model.state_dict(), model_save_path)
    print(f'\n训练好的 Transformer 模型已保存到 {model_save_path}')
except Exception as e:
    print(f"保存 PyTorch 模型时出错: {e}")

print("\nTransformer 脚本执行完毕。")