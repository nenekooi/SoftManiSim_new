# -*- coding: utf-8 -*-
# 导入必要的库
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import random
import joblib 
import os

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
DATA_FILE_PATH = 'D:/data/save_data_new/-x.xlsx' 

#    定义保存 LSTM 模型和 scalers 的目录
MODEL_SAVE_DIR = 'D:/data/lstm_residual_model_true' 
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- 4. 加载和预处理数据 ---
print(f"正在从以下路径加载数据: {DATA_FILE_PATH}")
if not os.path.exists(DATA_FILE_PATH):
    print(f"[错误] 数据文件未找到: {DATA_FILE_PATH}")
    exit()
try:
    df = pd.read_excel(DATA_FILE_PATH, engine='openpyxl')
    print(f"数据加载成功。形状: {df.shape}")
    # 检查必需的列
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
# --- !!! 确保使用正确的残差计算公式 !!! ---
df['residual_X'] = df['X_real_mm'] - df['sim_X_mm'] # <<< 修正 X 残差计算
df['residual_Y'] = df['Y_real_mm'] - df['sim_Y_mm']
df['residual_Z'] = df['Z_real_mm'] - df['sim_Z_mm']

# 处理原始仿真数据或真实数据中的 NaN (这些行无法用于计算残差或作为输入)
initial_rows = len(df)
df = df.dropna(subset=required_cols).copy() # 删除任何包含 NaN 的行
if len(df) < initial_rows:
    print(f"[警告] 原始数据中包含 NaN，已删除 {initial_rows - len(df)} 行。剩余 {len(df)} 行。")
if len(df) == 0:
    print("[错误] 没有有效的训练数据。")
    exit()

# --- 6. 定义输入和目标 ---
input_features = df[['cblen1_mm', 'cblen2_mm', 'cblen3_mm']].values
target_residuals = df[['residual_X', 'residual_Y', 'residual_Z']].values

print(f"输入特征形状: {input_features.shape}")
print(f"目标残差形状: {target_residuals.shape}")

# --- 7. 划分训练集和测试集 ---
# !!! 重要： shuffle=False 适用于时间序列数据。如果你的数据点是独立的，应设为 True !!!
# 假设数据有顺序性，保持 shuffle=False
split_ratio = 0.1 # 10% 测试集
split_index = int(len(input_features) * (1 - split_ratio))

X_train_raw = input_features[:split_index]
X_test_raw = input_features[split_index:]
y_train_raw = target_residuals[:split_index]
y_test_raw = target_residuals[split_index:]

# 保留测试集对应的原始仿真和真实坐标，用于后续绘图
sim_test_raw = df[['sim_X_mm', 'sim_Y_mm', 'sim_Z_mm']].iloc[split_index:].values
real_test_raw = df[['X_real_mm', 'Y_real_mm', 'Z_real_mm']].iloc[split_index:].values

print(f"训练集大小: {len(X_train_raw)}")
print(f"测试集大小: {len(X_test_raw)}")

# --- 8. 数据归一化 (使用 MinMaxScaler) ---
input_scaler = MinMaxScaler()
residual_scaler = MinMaxScaler() # 用于残差的 Scaler

# 仅用训练集拟合 Scaler
X_train = input_scaler.fit_transform(X_train_raw)
y_train = residual_scaler.fit_transform(y_train_raw)

# 应用到测试集
X_test = input_scaler.transform(X_test_raw)
y_test = residual_scaler.transform(y_test_raw)

# --- 保存 Scalers ---
x_scaler_path = os.path.join(MODEL_SAVE_DIR, 'lstm_x_scaler.joblib')
y_scaler_path = os.path.join(MODEL_SAVE_DIR, 'lstm_residual_scaler.joblib')
joblib.dump(input_scaler, x_scaler_path)
joblib.dump(residual_scaler, y_scaler_path)
print(f"输入 Scaler 保存至: {x_scaler_path}")
print(f"残差 Scaler 保存至: {y_scaler_path}")

# --- 9. 创建序列数据 ---
seq_length = 30  # LSTM 回看的步长 (可调整)
print(f"创建长度为 {seq_length} 的序列...")

def create_sequences(inputs, targets, seq_length):
    in_seqs = []
    out_seqs = []
    # 输入是 inputs[i:i+seq_length]
    # 输出是 targets[i+seq_length-1] (序列最后一个点对应的目标)
    # 或者 targets[i+seq_length] (预测序列之后下一个点的目标) - 原脚本是这个逻辑
    if len(inputs) <= seq_length:
        print("[警告] 数据长度不足以创建序列。")
        return np.array([]), np.array([])
    for i in range(len(inputs) - seq_length):
        in_seqs.append(inputs[i:i+seq_length])
        out_seqs.append(targets[i+seq_length]) # 预测序列后下一个点的目标
    if not in_seqs: # 如果循环没有产生序列
        print("[警告] 未能创建任何序列。")
        return np.array([]), np.array([])
    return np.array(in_seqs), np.array(out_seqs)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)

# 调整测试集原始坐标以匹配序列输出
# y_test_seq 对应的是原始索引 seq_length 到 len(X_test)-1 的目标
# 所以对应的 sim 和 real 坐标也应该从 seq_length 开始
if len(X_test_seq) > 0:
    sim_test_aligned = sim_test_raw[seq_length:]
    real_test_aligned = real_test_raw[seq_length:]
    # 确保长度匹配
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

batch_size = 64 # 可调整

# 检查是否有数据用于创建 DataLoader
if len(X_train_seq) == 0:
    print("[错误] 无法创建训练 DataLoader，没有训练序列。")
    exit()
if len(X_test_seq) == 0:
     print("[警告] 无法创建测试 DataLoader，没有测试序列。")
     # 可以选择退出或继续只训练

train_dataset = RobotDataset(X_train_seq, y_train_seq)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) # shuffle=False 保留序列性

if len(X_test_seq) > 0:
    test_dataset = RobotDataset(X_test_seq, y_test_seq)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
else:
    test_loader = None

# --- 11. 定义带注意力机制的 LSTM 模型 (与原脚本相同) ---
class Attention(nn.Module):
    # ... (代码与 LSTM_attention.py 中相同) ...
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size) # 修改: 输入维度改为 hidden*2
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / np.sqrt(self.v.size(0)) 
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        # hidden: (num_layers, batch_size, hidden_size) -> 取最后一层 (batch_size, hidden_size)
        # encoder_outputs: (batch_size, seq_length, hidden_size)
        seq_len = encoder_outputs.size(1)
        last_hidden = hidden[-1] # 取最后一层的隐藏状态
        last_hidden_expanded = last_hidden.unsqueeze(1).repeat(1, seq_len, 1) # (batch_size, seq_len, hidden_size)

        # 将隐藏状态和编码器输出拼接或相加后输入 attn 层
        attn_input = torch.cat((last_hidden_expanded, encoder_outputs), dim=2) # (batch_size, seq_len, hidden_size * 2)
        # 或者 attn_input = last_hidden_expanded + encoder_outputs

        energy = torch.tanh(self.attn(attn_input)) # (batch_size, seq_len, hidden_size)

        # 计算注意力权重
        energy = energy.transpose(1, 2) # (batch_size, hidden_size, seq_len)
        v_expanded = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1) # (batch_size, 1, hidden_size)
        attn_scores = torch.bmm(v_expanded, energy).squeeze(1) # (batch_size, seq_len)
        attn_weights = torch.softmax(attn_scores, dim=1) # (batch_size, seq_len)

        # 计算上下文向量
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs) # (batch_size, 1, hidden_size)
        context = context.squeeze(1) # (batch_size, hidden_size)

        return context, attn_weights


class LSTMWithAttention(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=2, output_size=3, dropout=0.2): # 增加 hidden_size 和 dropout
        super(LSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义 LSTM 层，增加 dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        # 定义注意力层
        self.attention = Attention(hidden_size)
        # 输出层 (上下文向量 -> 输出)
        self.fc = nn.Linear(hidden_size, output_size)
        # 可选：添加 Dropout 层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM 前向传播
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        # lstm_out: (batch_size, seq_length, hidden_size) - 所有时间步的输出
        # hn: (num_layers, batch_size, hidden_size) - 最后一个时间步的隐藏状态

        # 使用最后一个时间步的隐藏状态 hn 和所有时间步的输出 lstm_out 计算注意力
        # 注意：hn 的形状是 (num_layers, batch, hidden), attention 可能需要 (batch, hidden)
        context, attn_weights = self.attention(hn, lstm_out) # hn 作为 query, lstm_out 作为 values/keys

        # Dropout
        context = self.dropout(context)

        # 全连接层输出
        out = self.fc(context) # (batch_size, output_size)
        return out




# --- 12. 初始化模型、损失函数、优化器 ---
model = LSTMWithAttention(input_size=3, hidden_size=128, num_layers=2, output_size=3, dropout=0.2).to(device) # 使用调整后的参数
criterion = nn.MSELoss() # 均方误差损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Adam 优化器
# 可选：学习率调度器
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

print("\n模型结构:")
print(model)

# --- 13. 训练模型 ---
num_epochs = 100 # 可调整
train_losses = []
print("\n开始训练...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs_seq, targets_seq in train_loader:
        inputs_seq = inputs_seq.to(device)
        targets_seq = targets_seq.to(device)

        # 前向传播
        outputs = model(inputs_seq)
        loss = criterion(outputs, targets_seq)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        # 可选：梯度裁剪
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # 可选：使用学习率调度器
    # scheduler.step(avg_loss)

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')

print("训练完成。")

# --- 14. 绘制训练损失曲线 ---
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_losses, label='训练损失 (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('LSTM+Attention 模型训练损失曲线')
plt.legend()
plt.grid(True, linestyle=':')
loss_plot_path = os.path.join(MODEL_SAVE_DIR, 'lstm_training_loss.png')
plt.savefig(loss_plot_path)
print(f"训练损失曲线图已保存至: {loss_plot_path}")
# plt.show()

# --- 15. 测试模型 ---
if test_loader is None:
    print("\n无测试数据，跳过评估。")
else:
    print("\n开始评估模型...")
    model.eval()
    all_preds_scaled = []
    all_targets_scaled = []
    with torch.no_grad():
        for inputs_seq, targets_seq in test_loader:
            inputs_seq = inputs_seq.to(device)
            targets_seq = targets_seq.to(device) # 真实残差 (scaled)
            outputs = model(inputs_seq) # 预测残差 (scaled)
            all_preds_scaled.append(outputs.cpu().numpy())
            all_targets_scaled.append(targets_seq.cpu().numpy())

    # 拼接结果
    all_preds_scaled = np.concatenate(all_preds_scaled, axis=0)
    all_targets_scaled = np.concatenate(all_targets_scaled, axis=0) # 这是真实的 scaled 残差

    # 反归一化得到预测的残差 (毫米)
    predicted_residuals_mm = residual_scaler.inverse_transform(all_preds_scaled)
    # 反归一化得到真实的残差 (毫米) - 用于计算残差本身的 MAE
    true_residuals_mm = residual_scaler.inverse_transform(all_targets_scaled)

    # --- 评估残差预测的准确性 ---
    print("\n--- 残差预测评估 (毫米) ---")
    mae_residual_x = mean_absolute_error(true_residuals_mm[:, 0], predicted_residuals_mm[:, 0])
    mae_residual_y = mean_absolute_error(true_residuals_mm[:, 1], predicted_residuals_mm[:, 1])
    mae_residual_z = mean_absolute_error(true_residuals_mm[:, 2], predicted_residuals_mm[:, 2])
    avg_mae_residual = np.mean([mae_residual_x, mae_residual_y, mae_residual_z])
    print(f"残差 MAE: X={mae_residual_x:.3f}, Y={mae_residual_y:.3f}, Z={mae_residual_z:.3f}")
    print(f"平均残差 MAE: {avg_mae_residual:.3f} mm")

    # --- 计算修正后的仿真轨迹并评估 ---
    print("\n--- 最终轨迹评估 (毫米) ---")
    # 从之前保存的对齐数据中获取 sim 和 real 坐标
    sim_test = sim_test_aligned
    real_test = real_test_aligned

    # 检查 sim_test 是否包含 NaN
    nan_in_sim = np.isnan(sim_test).any(axis=1)
    if nan_in_sim.any():
        print(f"[警告] {nan_in_sim.sum()} 个测试样本的原始仿真坐标包含 NaN，将在计算修正轨迹和 MAE 时跳过。")
        valid_indices = ~nan_in_sim
        # 筛选所有相关数组
        sim_test = sim_test[valid_indices]
        real_test = real_test[valid_indices]
        predicted_residuals_mm = predicted_residuals_mm[valid_indices]
        # 如果需要绘制对比图，原始索引也需要筛选
        # test_indices = np.arange(len(sim_test_aligned))[valid_indices]

    # 计算修正后的轨迹
    corrected_sim_mm = sim_test + predicted_residuals_mm

    # 计算修正后轨迹与真实轨迹的 MAE
    mae_corrected_x = mean_absolute_error(real_test[:, 0], corrected_sim_mm[:, 0])
    mae_corrected_y = mean_absolute_error(real_test[:, 1], corrected_sim_mm[:, 1])
    mae_corrected_z = mean_absolute_error(real_test[:, 2], corrected_sim_mm[:, 2])
    mae_corrected_3d = np.mean(np.linalg.norm(real_test - corrected_sim_mm, axis=1))
    print(f"修正后轨迹 MAE: X={mae_corrected_x:.3f}, Y={mae_corrected_y:.3f}, Z={mae_corrected_z:.3f}")
    print(f"修正后轨迹 Overall 3D MAE: {mae_corrected_3d:.3f} mm")

    # --- 16. 绘制最终轨迹对比图 ---
    print("\n绘制最终轨迹对比图...")
    plot_indices = np.arange(len(real_test)) # 使用有效数据的索引

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    plot_title_final = f'真实轨迹 vs 修正后仿真轨迹 (LSTM+Attn, 3D MAE={mae_corrected_3d:.3f} mm)'

    # X 轴
    axs[0].plot(plot_indices, real_test[:, 0], label='真实 X (mm)', color='darkorange', linewidth=1.5)
    axs[0].plot(plot_indices, corrected_sim_mm[:, 0], label=f'修正后仿真 X (MAE={mae_corrected_x:.2f}mm)', color='royalblue', linestyle='--', linewidth=1)
    axs[0].set_ylabel('X 坐标 (mm)')
    axs[0].set_title('X 轴坐标对比')
    axs[0].legend()
    axs[0].grid(True, linestyle=':')

    # Y 轴
    axs[1].plot(plot_indices, real_test[:, 1], label='真实 Y (mm)', color='darkorange', linewidth=1.5)
    axs[1].plot(plot_indices, corrected_sim_mm[:, 1], label=f'修正后仿真 Y (MAE={mae_corrected_y:.2f}mm)', color='royalblue', linestyle='--', linewidth=1)
    axs[1].set_ylabel('Y 坐标 (mm)')
    axs[1].set_title('Y 轴坐标对比')
    axs[1].legend()
    axs[1].grid(True, linestyle=':')

    # Z 轴
    axs[2].plot(plot_indices, real_test[:, 2], label='真实 Z (mm)', color='darkorange', linewidth=1.5)
    axs[2].plot(plot_indices, corrected_sim_mm[:, 2], label=f'修正后仿真 Z (MAE={mae_corrected_z:.2f}mm)', color='royalblue', linestyle='--', linewidth=1)
    axs[2].set_ylabel('Z 坐标 (mm)')
    axs[2].set_title('Z 轴坐标对比')
    axs[2].legend()
    axs[2].grid(True, linestyle=':')

    axs[2].set_xlabel('测试集样本序号 (Aligned Index)')
    fig.suptitle(plot_title_final, fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    comparison_plot_path = os.path.join(MODEL_SAVE_DIR, 'lstm_trajectory_comparison.png')
    try:
        plt.savefig(comparison_plot_path, dpi=300)
        print(f"最终轨迹对比图已保存至: {comparison_plot_path}")
    except Exception as e:
        print(f"保存对比图像时出错: {e}")
    plt.show()

# --- 17. 保存训练好的 PyTorch 模型 ---
model_save_path = os.path.join(MODEL_SAVE_DIR, 'lstm_residual_model.pth')
try:
    torch.save(model.state_dict(), model_save_path)
    print(f'\n训练好的 PyTorch 模型已保存到 {model_save_path}')
except Exception as e:
    print(f"保存 PyTorch 模型时出错: {e}")

print("\n脚本执行完毕。")