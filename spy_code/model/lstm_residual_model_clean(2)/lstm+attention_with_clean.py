# -*- coding: utf-8 -*-
# 导入必要的库
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split # 保留以备将来使用，当前未使用
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error # r2_score 保留
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

set_seed(66)

# --- 2. 设备配置 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# --- 3. 配置: 修改为路径 ---
DATA_FILE_PATH = 'D:/data/save_data/parameter_clean1.xlsx' # 请确保这个文件是由 main_new.py 生成的
OUTPUT_EXCEL_RESULTS_PATH = os.path.join(os.path.dirname(DATA_FILE_PATH), 'lstm_test_results_output.xlsx') # 新增：输出Excel文件名

# 定义保存 LSTM 模型和 scalers 的目录
MODEL_SAVE_DIR = 'D:/data/save_model/lstm_residual_model_clean(2)' # 可以为新模型更改目录名
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
df['residual_X'] = df['X_real_mm'] - df['sim_X_mm']
df['residual_Y'] = df['Y_real_mm'] - df['sim_Y_mm']
df['residual_Z'] = df['Z_real_mm'] - df['sim_Z_mm']

initial_rows = len(df)
df = df.dropna(subset=required_cols + ['residual_X', 'residual_Y', 'residual_Z']).copy()
if len(df) < initial_rows:
    print(f"[警告] 原始数据或计算出的残差中包含 NaN，已删除 {initial_rows - len(df)} 行。剩余 {len(df)} 行。")
if len(df) == 0:
    print("[错误] 没有有效的训练数据。")
    exit()

# --- 6. 定义输入和目标 ---
input_feature_names = ['cblen1_mm', 'cblen2_mm', 'cblen3_mm', 'sim_X_mm', 'sim_Y_mm', 'sim_Z_mm']
input_features = df[input_feature_names].values
target_residuals = df[['residual_X', 'residual_Y', 'residual_Z']].values

print(f"输入特征 ({len(input_feature_names)}个) 形状: {input_features.shape}")
print(f"目标残差形状: {target_residuals.shape}")

# --- 7. 划分训练集和测试集 ---
split_ratio = 0.1
split_index = int(len(input_features) * (1 - split_ratio))

X_train_raw = input_features[:split_index]
X_test_raw = input_features[split_index:] # 这个是后续要对齐并保存的输入特征
y_train_raw = target_residuals[:split_index]
y_test_raw = target_residuals[split_index:]

sim_test_raw_for_correction = df[['sim_X_mm', 'sim_Y_mm', 'sim_Z_mm']].iloc[split_index:].values
real_test_raw = df[['X_real_mm', 'Y_real_mm', 'Z_real_mm']].iloc[split_index:].values
# ***** 新增：保留测试集的原始绳长，用于输出Excel *****
cable_lengths_test_raw = df[['cblen1_mm', 'cblen2_mm', 'cblen3_mm']].iloc[split_index:].values


print(f"训练集大小: {len(X_train_raw)}")
print(f"测试集大小: {len(X_test_raw)}")

# --- 8. 数据归一化 ---
input_scaler = MinMaxScaler()
residual_scaler = MinMaxScaler()

X_train = input_scaler.fit_transform(X_train_raw)
y_train = residual_scaler.fit_transform(y_train_raw)
X_test = input_scaler.transform(X_test_raw)
y_test = residual_scaler.transform(y_test_raw)

x_scaler_path = os.path.join(MODEL_SAVE_DIR, 'lstm_x_scaler_6features.joblib')
y_scaler_path = os.path.join(MODEL_SAVE_DIR, 'lstm_residual_scaler.joblib')
joblib.dump(input_scaler, x_scaler_path)
joblib.dump(residual_scaler, y_scaler_path)
print(f"输入 Scaler (6 特征) 保存至: {x_scaler_path}")
print(f"残差 Scaler 保存至: {y_scaler_path}")

# --- 9. 创建序列数据 ---
seq_length = 30
print(f"创建长度为 {seq_length} 的序列...")

def create_sequences(inputs, targets, seq_length):
    in_seqs = []
    out_seqs = []
    if len(inputs) <= seq_length:
        print("[警告] 数据长度不足以创建序列。")
        return np.array([]), np.array([])
    for i in range(len(inputs) - seq_length):
        in_seqs.append(inputs[i:i+seq_length])
        out_seqs.append(targets[i+seq_length])
    if not in_seqs:
        print("[警告] 未能创建任何序列。")
        return np.array([]), np.array([])
    return np.array(in_seqs), np.array(out_seqs)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)

# 对齐所有用于评估和输出的测试集数据
if len(X_test_seq) > 0:
    sim_test_aligned_for_correction = sim_test_raw_for_correction[seq_length:]
    real_test_aligned = real_test_raw[seq_length:]
    # ***** 新增：对齐测试集的绳长数据 *****
    cable_lengths_test_aligned = cable_lengths_test_raw[seq_length:]


    min_len = min(len(y_test_seq), len(sim_test_aligned_for_correction), len(real_test_aligned), len(cable_lengths_test_aligned))
    y_test_seq = y_test_seq[:min_len]
    sim_test_aligned_for_correction = sim_test_aligned_for_correction[:min_len]
    real_test_aligned = real_test_aligned[:min_len]
    cable_lengths_test_aligned = cable_lengths_test_aligned[:min_len] # 对齐绳长

    print(f"序列化后，用于评估/绘图/输出的测试点数量: {min_len}")
else:
    print("[警告] 未能生成测试序列，无法进行评估和绘图。")
    sim_test_aligned_for_correction = np.array([])
    real_test_aligned = np.array([])
    cable_lengths_test_aligned = np.array([]) # 初始化为空


# --- 10. 自定义 Dataset 和 DataLoader ---
class RobotDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

batch_size = 128
if len(X_train_seq) == 0:
    print("[错误] 无法创建训练 DataLoader，没有训练序列。")
    exit()
train_dataset = RobotDataset(X_train_seq, y_train_seq)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = None
if len(X_test_seq) > 0:
    test_dataset = RobotDataset(X_test_seq, y_test_seq)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
else:
     print("[警告] 无法创建测试 DataLoader，没有测试序列。评估将被跳过。")

# --- 11. 定义带注意力机制的 LSTM 模型 ---
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / np.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)
    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        last_hidden = hidden[-1]
        last_hidden_expanded = last_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        attn_input = torch.cat((last_hidden_expanded, encoder_outputs), dim=2)
        energy = torch.tanh(self.attn(attn_input))
        energy = energy.transpose(1, 2)
        v_expanded = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        attn_scores = torch.bmm(v_expanded, energy).squeeze(1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)
        return context, attn_weights

class LSTMWithAttention(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, output_size=3, dropout=0.2):
        super(LSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        context, attn_weights = self.attention(hn, lstm_out)
        context = self.dropout(context)
        out = self.fc(context)
        return out

# --- 12. 初始化模型、损失函数、优化器 ---
model = LSTMWithAttention(input_size=len(input_feature_names), hidden_size=128, num_layers=2, output_size=3, dropout=0.2).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\n模型结构:")
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型总可训练参数: {total_params}")

# --- 13. 训练模型 ---
num_epochs = 150 # 保持100个epoch作为示例
train_losses = []
print("\n开始训练...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs_seq, targets_seq in train_loader:
        inputs_seq = inputs_seq.to(device)
        targets_seq = targets_seq.to(device)
        outputs = model(inputs_seq)
        loss = criterion(outputs, targets_seq)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
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
# plt.show() # 在服务器或非GUI环境运行时注释掉

# --- 15. 测试模型 ---
# ***** 初始化 corrected_sim_mm 以备后续使用 *****
corrected_sim_mm = np.array([]) # 初始化为空，如果测试不运行则保持为空

if test_loader is None or len(X_test_seq) == 0 :
    print("\n无测试数据或测试序列为空，跳过评估和结果保存。")
else:
    print("\n开始评估模型...")
    model.eval()
    all_preds_scaled = []
    all_targets_scaled = []
    with torch.no_grad():
        for inputs_seq, targets_seq_batch in test_loader:
            inputs_seq = inputs_seq.to(device)
            outputs_scaled_residuals = model(inputs_seq)
            all_preds_scaled.append(outputs_scaled_residuals.cpu().numpy())
            all_targets_scaled.append(targets_seq_batch.cpu().numpy())
    all_preds_scaled = np.concatenate(all_preds_scaled, axis=0)
    all_targets_scaled = np.concatenate(all_targets_scaled, axis=0)
    predicted_residuals_mm = residual_scaler.inverse_transform(all_preds_scaled)
    true_residuals_mm_for_eval = residual_scaler.inverse_transform(all_targets_scaled)

    print("\n--- 残差预测评估 (毫米) ---")
    mae_residual_x = mean_absolute_error(true_residuals_mm_for_eval[:, 0], predicted_residuals_mm[:, 0])
    mae_residual_y = mean_absolute_error(true_residuals_mm_for_eval[:, 1], predicted_residuals_mm[:, 1])
    mae_residual_z = mean_absolute_error(true_residuals_mm_for_eval[:, 2], predicted_residuals_mm[:, 2])
    avg_mae_residual = np.mean([mae_residual_x, mae_residual_y, mae_residual_z])
    print(f"残差 MAE: X={mae_residual_x:.3f}, Y={mae_residual_y:.3f}, Z={mae_residual_z:.3f}")
    print(f"平均残差 MAE: {avg_mae_residual:.3f} mm")

    print("\n--- 最终轨迹评估 (毫米) ---")
    sim_test_for_correction_mm = sim_test_aligned_for_correction
    real_test_mm = real_test_aligned

    nan_in_sim = np.isnan(sim_test_for_correction_mm).any(axis=1)
    valid_indices_for_eval = ~nan_in_sim
    if nan_in_sim.any():
        print(f"[警告] {nan_in_sim.sum()} 个测试样本的原始仿真坐标包含 NaN，将在计算修正轨迹和 MAE 时跳过。")
        sim_test_for_correction_mm = sim_test_for_correction_mm[valid_indices_for_eval]
        real_test_mm = real_test_mm[valid_indices_for_eval]
        predicted_residuals_mm = predicted_residuals_mm[valid_indices_for_eval]
        # ***** 新增：同样筛选对齐的绳长数据 *****
        if len(cable_lengths_test_aligned) > 0: # 确保 cable_lengths_test_aligned 不是空的
             cable_lengths_test_aligned = cable_lengths_test_aligned[valid_indices_for_eval]


    if len(predicted_residuals_mm) == 0 or len(sim_test_for_correction_mm) == 0:
        print("[错误] 没有有效数据点用于计算修正轨迹的评估。")
    else:
        corrected_sim_mm = sim_test_for_correction_mm + predicted_residuals_mm # 这个是最终结果
        mae_corrected_x = mean_absolute_error(real_test_mm[:, 0], corrected_sim_mm[:, 0])
        mae_corrected_y = mean_absolute_error(real_test_mm[:, 1], corrected_sim_mm[:, 1])
        mae_corrected_z = mean_absolute_error(real_test_mm[:, 2], corrected_sim_mm[:, 2])
        mae_corrected_3d = np.mean(np.linalg.norm(real_test_mm - corrected_sim_mm, axis=1))
        print(f"修正后轨迹 MAE: X={mae_corrected_x:.3f}, Y={mae_corrected_y:.3f}, Z={mae_corrected_z:.3f}")
        print(f"修正后轨迹 Overall 3D MAE: {mae_corrected_3d:.3f} mm")

        # --- 16. 绘制最终轨迹对比图 ---
        # (这部分绘图代码保持不变，因为它已经使用了 corrected_sim_mm, real_test_mm, sim_test_for_correction_mm)
        print("\n绘制最终轨迹对比图...")
        plot_indices = np.arange(len(real_test_mm))
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        plot_title_final = f'真实轨迹 vs 修正后仿真轨迹 (LSTM+Attn, 3D MAE={mae_corrected_3d:.3f} mm)'
        axs[0].plot(plot_indices, real_test_mm[:, 0], label='真实 X (mm)', color='darkorange', linewidth=1.5)
        axs[0].plot(plot_indices, corrected_sim_mm[:, 0], label=f'修正后仿真 X (MAE={mae_corrected_x:.2f}mm)', color='royalblue', linestyle='--', linewidth=1)
        axs[0].plot(plot_indices, sim_test_for_correction_mm[:, 0], label='原始仿真 X (mm)', color='green', linestyle=':', linewidth=0.8, alpha=0.7)
        axs[0].set_ylabel('X 坐标 (mm)'); axs[0].set_title('X 轴坐标对比'); axs[0].legend(); axs[0].grid(True, linestyle=':')
        axs[1].plot(plot_indices, real_test_mm[:, 1], label='真实 Y (mm)', color='darkorange', linewidth=1.5)
        axs[1].plot(plot_indices, corrected_sim_mm[:, 1], label=f'修正后仿真 Y (MAE={mae_corrected_y:.2f}mm)', color='royalblue', linestyle='--', linewidth=1)
        axs[1].plot(plot_indices, sim_test_for_correction_mm[:, 1], label='原始仿真 Y (mm)', color='green', linestyle=':', linewidth=0.8, alpha=0.7)
        axs[1].set_ylabel('Y 坐标 (mm)'); axs[1].set_title('Y 轴坐标对比'); axs[1].legend(); axs[1].grid(True, linestyle=':')
        axs[2].plot(plot_indices, real_test_mm[:, 2], label='真实 Z (mm)', color='darkorange', linewidth=1.5)
        axs[2].plot(plot_indices, corrected_sim_mm[:, 2], label=f'修正后仿真 Z (MAE={mae_corrected_z:.2f}mm)', color='royalblue', linestyle='--', linewidth=1)
        axs[2].plot(plot_indices, sim_test_for_correction_mm[:, 2], label='原始仿真 Z (mm)', color='green', linestyle=':', linewidth=0.8, alpha=0.7)
        axs[2].set_ylabel('Z 坐标 (mm)'); axs[2].set_title('Z 轴坐标对比'); axs[2].legend(); axs[2].grid(True, linestyle=':')
        axs[2].set_xlabel('测试集样本序号 (Aligned Index)')
        fig.suptitle(plot_title_final, fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])
        comparison_plot_path = os.path.join(MODEL_SAVE_DIR, 'lstm_trajectory_comparison.png')
        try:
            plt.savefig(comparison_plot_path, dpi=300)
            print(f"最终轨迹对比图已保存至: {comparison_plot_path}")
        except Exception as e:
            print(f"保存对比图像时出错: {e}")
        # plt.show() # 在服务器或非GUI环境运行时注释掉

# --- 新增：保存测试集结果到Excel ---
if len(corrected_sim_mm) > 0 and len(real_test_aligned) > 0 and len(sim_test_aligned_for_correction) > 0 and len(cable_lengths_test_aligned) > 0:
    print(f"\n准备将测试结果保存到Excel: {OUTPUT_EXCEL_RESULTS_PATH}")
    
    # 确保所有数组长度一致 (理论上在前面已经通过 min_len 和 valid_indices_for_eval 保证了)
    # 我们使用在评估部分最终筛选和对齐后的数据
    # real_test_mm, sim_test_for_correction_mm, corrected_sim_mm, cable_lengths_test_aligned, predicted_residuals_mm
    
    # 创建DataFrame
    results_df_data = {
        'cblen1_mm': cable_lengths_test_aligned[:, 0],
        'cblen2_mm': cable_lengths_test_aligned[:, 1],
        'cblen3_mm': cable_lengths_test_aligned[:, 2],
        'X_real_mm': real_test_mm[:, 0],
        'Y_real_mm': real_test_mm[:, 1],
        'Z_real_mm': real_test_mm[:, 2],
        'sim_X_raw_mm': sim_test_for_correction_mm[:, 0],
        'sim_Y_raw_mm': sim_test_for_correction_mm[:, 1],
        'sim_Z_raw_mm': sim_test_for_correction_mm[:, 2],
        'predicted_residual_X_mm': predicted_residuals_mm[:, 0],
        'predicted_residual_Y_mm': predicted_residuals_mm[:, 1],
        'predicted_residual_Z_mm': predicted_residuals_mm[:, 2],
        'X_corrected_mm': corrected_sim_mm[:, 0],
        'Y_corrected_mm': corrected_sim_mm[:, 1],
        'Z_corrected_mm': corrected_sim_mm[:, 2]
    }
    
    # 计算每个点的误差，也加入到Excel中
    error_x_col = real_test_mm[:, 0] - corrected_sim_mm[:, 0]
    error_y_col = real_test_mm[:, 1] - corrected_sim_mm[:, 1]
    error_z_col = real_test_mm[:, 2] - corrected_sim_mm[:, 2]
    error_3d_col = np.linalg.norm(real_test_mm - corrected_sim_mm, axis=1)

    results_df_data['Error_X_mm'] = error_x_col
    results_df_data['Error_Y_mm'] = error_y_col
    results_df_data['Error_Z_mm'] = error_z_col
    results_df_data['Error_3D_mm'] = error_3d_col

    results_df = pd.DataFrame(results_df_data)
    
    try:
        results_df.to_excel(OUTPUT_EXCEL_RESULTS_PATH, index=False, engine='openpyxl')
        print(f"测试集结果已成功保存到: {OUTPUT_EXCEL_RESULTS_PATH}")
    except Exception as e:
        print(f"保存测试集结果Excel文件时出错: {e}")
else:
    print("\n没有有效的测试结果可供保存到Excel。")

# --- 17. 保存训练好的 PyTorch 模型 ---
model_save_path = os.path.join(MODEL_SAVE_DIR, 'lstm_residual_model.pth') # 恢复原始模型名
try:
    torch.save(model.state_dict(), model_save_path)
    print(f'\n训练好的 PyTorch 模型已保存到 {model_save_path}')
except Exception as e:
    print(f"保存 PyTorch 模型时出错: {e}")

print("\n脚本执行完毕。")