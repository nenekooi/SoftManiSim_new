# -*- coding: utf-8 -*-
# 新脚本：load_lstm_and_predict_to_excel.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader # DataLoader 可能不需要，除非你想分批预测
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import random # 为了 set_seed

# --- 中文显示设置 (如果需要在脚本中绘图或打印，可以保留) ---
# try:
#     plt.rcParams['font.family'] = ['SimHei']
#     plt.rcParams['axes.unicode_minus'] = False
# except Exception:
#     pass

# --- 1. 从 LSTM_attention_model.py 复制模型定义 ---
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
    def __init__(self, input_size=3, hidden_size=128, num_layers=2, output_size=3, dropout=0.2):
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

# --- 2. 设置随机种子 (保持和训练时一致可能有助于某些情况，但对推理不是必须) ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# --- 3. 设备配置 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# --- 4. 配置路径 ---
# 原始数据文件路径 (与 LSTM_attention_model.py 中的一致)
DATA_FILE_PATH = 'D:/data/save_data/aaa2(u_new_5,cab=0.035,k=-2,a=0.8).xlsx'

# 模型和 Scalers 保存的目录 (与 LSTM_attention_model.py 中的 MODEL_SAVE_DIR 一致)
MODEL_SAVE_DIR = 'D:/data/save_model/lstm_residual_model_true_newdata' # <<<< 确保这是你保存模型和scaler的正确目录

# 已训练模型的具体路径
MODEL_LOAD_PATH = os.path.join(MODEL_SAVE_DIR, 'lstm_residual_model.pth') # <<<< 你的.pth模型文件

# Scalers 的具体路径
X_SCALER_PATH = os.path.join(MODEL_SAVE_DIR, 'lstm_x_scaler.joblib')
RESIDUAL_SCALER_PATH = os.path.join(MODEL_SAVE_DIR, 'lstm_residual_scaler.joblib')

# 输出 Excel 文件的路径和名称
OUTPUT_EXCEL_DIR = MODEL_SAVE_DIR # 可以和模型保存在一起，或指定新目录
os.makedirs(OUTPUT_EXCEL_DIR, exist_ok=True)
OUTPUT_EXCEL_FILENAME = 'lstm_test_set_inference_results.xlsx'
OUTPUT_EXCEL_PATH = os.path.join(OUTPUT_EXCEL_DIR, OUTPUT_EXCEL_FILENAME)

# --- 5. 加载和预处理数据 (大部分复用 LSTM_attention_model.py 的逻辑) ---
print(f"--- 正在从以下路径加载原始数据: {DATA_FILE_PATH} ---")
if not os.path.exists(DATA_FILE_PATH):
    print(f"[错误] 数据文件未找到: {DATA_FILE_PATH}")
    exit()
try:
    df = pd.read_excel(DATA_FILE_PATH, engine='openpyxl')
    print(f"原始数据加载成功。形状: {df.shape}")
except Exception as e:
    print(f"加载原始数据时出错: {e}")
    exit()

# 计算残差
required_cols = ['cblen1_mm', 'cblen2_mm', 'cblen3_mm',
                 'X_real_mm', 'Y_real_mm', 'Z_real_mm',
                 'sim_X_mm', 'sim_Y_mm', 'sim_Z_mm']
if not all(col in df.columns for col in required_cols):
    missing = [col for col in required_cols if col not in df.columns]
    print(f"[错误] 原始数据文件缺少必需的列: {missing}")
    exit()

df['residual_X'] = df['X_real_mm'] - df['sim_X_mm']
df['residual_Y'] = df['Y_real_mm'] - df['sim_Y_mm']
df['residual_Z'] = df['Z_real_mm'] - df['sim_Z_mm']

initial_rows = len(df)
df_cleaned = df.dropna(subset=required_cols + ['residual_X', 'residual_Y', 'residual_Z']).copy()
if len(df_cleaned) < initial_rows:
    print(f"[警告] 原始数据中包含 NaN，已删除 {initial_rows - len(df_cleaned)} 行。剩余 {len(df_cleaned)} 行。")
if len(df_cleaned) == 0:
    print("[错误] 清理 NaN 后没有有效的原始数据。")
    exit()

input_features_raw = df_cleaned[['cblen1_mm', 'cblen2_mm', 'cblen3_mm']].values
target_residuals_raw = df_cleaned[['residual_X', 'residual_Y', 'residual_Z']].values # 这是真实的残差

# 划分训练集和测试集 (必须与训练时完全一致，以获取正确的测试集)
# !!! 使用与 LSTM_attention_model.py 中完全相同的 split_ratio !!!
split_ratio = 0.1 # 10% 测试集
split_index = int(len(input_features_raw) * (1 - split_ratio))

# 我们只关心测试集
X_test_raw = input_features_raw[split_index:]
y_test_raw_residuals = target_residuals_raw[split_index:] # 测试集的真实残差 (未归一化)

# 获取测试集对应的原始仿真坐标和真实坐标
sim_test_raw_coords = df_cleaned[['sim_X_mm', 'sim_Y_mm', 'sim_Z_mm']].iloc[split_index:].values
real_test_raw_coords = df_cleaned[['X_real_mm', 'Y_real_mm', 'Z_real_mm']].iloc[split_index:].values

print(f"提取的原始测试集大小: {len(X_test_raw)}")
if len(X_test_raw) == 0:
    print("[错误] 未能提取任何测试数据。请检查 split_ratio 和数据量。")
    exit()

# --- 6. 加载 Scalers ---
print("--- 正在加载 Scalers ---")
if not os.path.exists(X_SCALER_PATH) or not os.path.exists(RESIDUAL_SCALER_PATH):
    print(f"[错误] Scaler 文件未找到。请确保路径正确: {X_SCALER_PATH}, {RESIDUAL_SCALER_PATH}")
    exit()
try:
    input_scaler = joblib.load(X_SCALER_PATH)
    residual_scaler = joblib.load(RESIDUAL_SCALER_PATH)
    print("Scalers 加载成功。")
except Exception as e:
    print(f"加载 Scalers 时发生错误: {e}")
    exit()

# --- 7. 预处理测试数据并创建序列 (与训练时一致) ---
print("--- 正在预处理测试数据并创建序列 ---")
X_test_scaled = input_scaler.transform(X_test_raw)
# y_test_scaled_residuals = residual_scaler.transform(y_test_raw_residuals) # 仅用于验证，模型输出的是scaled residual

# !!! 使用与 LSTM_attention_model.py 中完全相同的 seq_length !!!
seq_length = 30
print(f"创建长度为 {seq_length} 的测试序列...")

def create_sequences_input_only(inputs, seq_length): # 只需要输入序列
    in_seqs = []
    if len(inputs) <= seq_length:
        print("[警告] 测试数据长度不足以创建序列。")
        return np.array([])
    for i in range(len(inputs) - seq_length):
        in_seqs.append(inputs[i : i + seq_length])
    if not in_seqs:
        print("[警告] 未能创建任何测试序列。")
        return np.array([])
    return np.array(in_seqs)

X_test_seq = create_sequences_input_only(X_test_scaled, seq_length)

if len(X_test_seq) == 0:
    print("[错误] 未能生成测试序列，无法进行预测。")
    exit()

# 对齐原始坐标和真实残差 (因为序列化会减少数据开头的部分)
# X_test_seq[i] 对应的原始数据点是 X_test_raw[i+seq_length]
# 所以，对齐的真实坐标应该是 real_test_raw_coords 从第 seq_length 个点开始
# 对齐的原始仿真坐标应该是 sim_test_raw_coords 从第 seq_length 个点开始
# 对齐的真实残差应该是 y_test_raw_residuals 从第 seq_length 个点开始

num_predictions = len(X_test_seq)
real_test_aligned_coords = real_test_raw_coords[seq_length : seq_length + num_predictions]
sim_test_aligned_coords = sim_test_raw_coords[seq_length : seq_length + num_predictions]
true_residuals_aligned_mm = y_test_raw_residuals[seq_length : seq_length + num_predictions] # 真实的残差，未归一化

print(f"生成的测试序列数量: {num_predictions}")
print(f"对齐后的真实/仿真坐标数量: {len(real_test_aligned_coords)}")

if len(real_test_aligned_coords) != num_predictions:
    print("[错误] 对齐后的坐标数量与预测数量不匹配，请检查序列创建逻辑。")
    # 通常这意味着原始测试数据太少，无法在偏移seq_length后仍然覆盖所有序列
    # 修正：确保截取长度不超过可用长度
    min_len = min(num_predictions, len(real_test_raw_coords) - seq_length, len(sim_test_raw_coords) - seq_length, len(y_test_raw_residuals) - seq_length)
    if min_len < num_predictions :
        print(f"[警告] 由于数据长度限制，实际可用的对齐点数为 {min_len}，将截断预测序列。")
        X_test_seq = X_test_seq[:min_len]
        num_predictions = min_len
        real_test_aligned_coords = real_test_raw_coords[seq_length : seq_length + num_predictions]
        sim_test_aligned_coords = sim_test_raw_coords[seq_length : seq_length + num_predictions]
        true_residuals_aligned_mm = y_test_raw_residuals[seq_length : seq_length + num_predictions]


# --- 8. 加载已训练的模型 ---
print("--- 正在加载已训练的 LSTM+Attention 模型 ---")
# 与 LSTM_attention_model.py 中模型参数一致
model_params = {
    'input_size': X_test_seq.shape[2], # 应该为3
    'hidden_size': 128,
    'num_layers': 2,
    'output_size': 3, # 输出残差的维度
    'dropout': 0.2
}
model = LSTMWithAttention(**model_params).to(device)

if not os.path.exists(MODEL_LOAD_PATH):
    print(f"[错误] 模型文件未找到: {MODEL_LOAD_PATH}")
    exit()
try:
    model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device)) # map_location 确保在不同设备上也能加载
    model.eval() # 设置为评估模式
    print("模型加载成功并设置为评估模式。")
except Exception as e:
    print(f"加载模型时发生错误: {e}")
    exit()

# --- 9. 模型推理（预测残差） ---
print("--- 正在使用加载的模型进行预测 ---")
predicted_residuals_scaled_list = []
with torch.no_grad(): # 推理时不需要计算梯度
    # 如果测试集很大，可以考虑分批处理，但这里我们一次性处理
    inputs_tensor = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
    outputs_tensor = model(inputs_tensor)
    predicted_residuals_scaled = outputs_tensor.cpu().numpy()

# --- 10. 反归一化预测的残差 ---
predicted_residuals_mm = residual_scaler.inverse_transform(predicted_residuals_scaled)
print(f"预测的残差形状 (mm): {predicted_residuals_mm.shape}")

# --- 11. 计算修正后的仿真坐标 ---
print("--- 正在计算修正后的仿真坐标 ---")
if len(sim_test_aligned_coords) != len(predicted_residuals_mm):
    print(f"[错误] 对齐的原始仿真坐标数量 ({len(sim_test_aligned_coords)}) 与预测的残差数量 ({len(predicted_residuals_mm)}) 不匹配。")
    # 进一步确保截断
    min_len_final = min(len(sim_test_aligned_coords), len(predicted_residuals_mm), len(real_test_aligned_coords), len(true_residuals_aligned_mm))
    sim_test_aligned_coords = sim_test_aligned_coords[:min_len_final]
    predicted_residuals_mm = predicted_residuals_mm[:min_len_final]
    real_test_aligned_coords = real_test_aligned_coords[:min_len_final]
    true_residuals_aligned_mm = true_residuals_aligned_mm[:min_len_final]
    print(f"最终用于保存的数据点数: {min_len_final}")


corrected_sim_coords_mm = sim_test_aligned_coords + predicted_residuals_mm
print(f"修正后的仿真坐标形状: {corrected_sim_coords_mm.shape}")


# --- 12. 整理数据并保存到 Excel ---
print(f"--- 正在将结果保存到 Excel 文件: {OUTPUT_EXCEL_PATH} ---")
if len(real_test_aligned_coords) > 0:
    output_data_dict = {
        'X_real_mm': real_test_aligned_coords[:, 0],
        'Y_real_mm': real_test_aligned_coords[:, 1],
        'Z_real_mm': real_test_aligned_coords[:, 2],
        'sim_X_mm_aligned': sim_test_aligned_coords[:, 0],
        'sim_Y_mm_aligned': sim_test_aligned_coords[:, 1],
        'sim_Z_mm_aligned': sim_test_aligned_coords[:, 2],
        'true_residual_X_mm': true_residuals_aligned_mm[:, 0], # 真实的残差
        'true_residual_Y_mm': true_residuals_aligned_mm[:, 1],
        'true_residual_Z_mm': true_residuals_aligned_mm[:, 2],
        'predicted_residual_X_mm': predicted_residuals_mm[:, 0],
        'predicted_residual_Y_mm': predicted_residuals_mm[:, 1],
        'predicted_residual_Z_mm': predicted_residuals_mm[:, 2],
        'corrected_sim_X_mm': corrected_sim_coords_mm[:, 0],
        'corrected_sim_Y_mm': corrected_sim_coords_mm[:, 1],
        'corrected_sim_Z_mm': corrected_sim_coords_mm[:, 2]
    }
    output_df = pd.DataFrame(output_data_dict)
    try:
        output_df.to_excel(OUTPUT_EXCEL_PATH, index=False, engine='openpyxl')
        print(f"结果已成功保存。包含 {len(output_df)} 行数据。")
    except Exception as e:
        print(f"[错误] 保存到 Excel 时发生错误: {e}")
else:
    print("[警告] 没有最终的对齐数据可供保存。")

print("--- 脚本执行完毕 ---")