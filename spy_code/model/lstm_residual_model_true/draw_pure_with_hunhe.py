import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 中文显示设置 ---
try:
    plt.rcParams['font.family'] = ['SimHei'] # 或者 'Microsoft YaHei', 'Arial Unicode MS'
    plt.rcParams['axes.unicode_minus'] = False
    print("[信息] 尝试设置字体以支持中文。")
except Exception as e:
    print(f"[警告] 设置中文字体失败: {e}。部分标签可能无法正确显示。")
    print("[提示] 如果中文显示不正确，请确保你的系统中安装了 SimHei 或其他中文字体。")

# --- 1. 配置参数 ---
# 输入的 Excel 文件路径 (由上一个脚本生成)
# MODEL_SAVE_DIR 应该与 load_lstm_and_predict_to_excel.py 中的一致
MODEL_OUTPUT_DIR = 'D:/data/save_model/lstm_residual_model_true_newdata' # <<<< 确保这是你保存 lstm_test_set_inference_results.xlsx 的正确目录
# EXCEL_FILE_NAME = 'lstm_test_set_inference_results.xlsx'
EXCEL_FILE_NAME = 'lstm_test_set_inference_results.xlsx'
INPUT_EXCEL_PATH = os.path.join(MODEL_OUTPUT_DIR, EXCEL_FILE_NAME)

# 输出图像的保存目录
FIGURES_SAVE_DIR = 'D:/data/save_model/lstm_residual_model_true_newdata' # 你可以指定一个新的目录
os.makedirs(FIGURES_SAVE_DIR, exist_ok=True)
OUTPUT_PLOT_FILENAME = 'comparison_error_boxplot_final.png'
OUTPUT_PLOT_PATH = os.path.join(FIGURES_SAVE_DIR, OUTPUT_PLOT_FILENAME)

# --- 2. 加载已处理好的数据 ---
print(f"--- 正在加载已处理的测试集结果数据来源: {INPUT_EXCEL_PATH} ---")
if not os.path.exists(INPUT_EXCEL_PATH):
    print(f"[错误] 文件未找到: {INPUT_EXCEL_PATH}")
    print("[提示] 请先运行 'load_lstm_and_predict_to_excel.py' 脚本生成此文件。")
    exit()
try:
    df_results = pd.read_excel(INPUT_EXCEL_PATH, engine='openpyxl')
    print(f"数据加载成功。数据形状: {df_results.shape}")
except Exception as e:
    print(f"加载 Excel 文件时发生错误: {e}")
    exit()

# --- 3. 提取所需的坐标列 ---
# 确保列名与 'load_lstm_and_predict_to_excel.py' 输出的 Excel 中的列名完全一致
required_cols_for_plot = [
    'X_real_mm', 'Y_real_mm', 'Z_real_mm',                      # 真实坐标
    'sim_X_mm_aligned', 'sim_Y_mm_aligned', 'sim_Z_mm_aligned', # 对齐的纯物理仿真
    'corrected_sim_X_mm', 'corrected_sim_Y_mm', 'corrected_sim_Z_mm' # NN修正后的仿真
]
if not all(col in df_results.columns for col in required_cols_for_plot):
    missing = [col for col in required_cols_for_plot if col not in df_results.columns]
    print(f"[错误] 输入的 Excel 文件缺少必要的列: {missing}")
    print(f"  Excel 文件中的可用列: {df_results.columns.tolist()}")
    exit()

# 提取数据为 NumPy 数组
real_coords = df_results[['X_real_mm', 'Y_real_mm', 'Z_real_mm']].values
physics_sim_coords = df_results[['sim_X_mm_aligned', 'sim_Y_mm_aligned', 'sim_Z_mm_aligned']].values
nn_corrected_coords = df_results[['corrected_sim_X_mm', 'corrected_sim_Y_mm', 'corrected_sim_Z_mm']].values

if not (len(real_coords) == len(physics_sim_coords) == len(nn_corrected_coords)):
    print("[错误] 加载的数据列长度不一致，无法进行比较。请检查Excel文件。")
    exit()
if len(real_coords) == 0:
    print("[错误] 加载的数据为空，无法绘图。")
    exit()

print(f"用于绘图的有效数据点数量: {len(real_coords)}")

# --- 4. 计算两组模型的误差 ---
def calculate_model_errors(real, sim, model_name):
    """辅助函数计算给定真实值和仿真值的误差"""
    abs_err_x = np.abs(sim[:, 0] - real[:, 0])
    abs_err_y = np.abs(sim[:, 1] - real[:, 1])
    abs_err_z = np.abs(sim[:, 2] - real[:, 2])
    pos_err = np.sqrt(np.sum((sim - real)**2, axis=1))

    print(f"\n--- {model_name} 误差统计 ---")
    print(f"  MAE X: {np.mean(abs_err_x):.3f}, Y: {np.mean(abs_err_y):.3f}, Z: {np.mean(abs_err_z):.3f} mm")
    print(f"  平均三维位置误差: {np.mean(pos_err):.3f} mm")
    print(f"  最大三维位置误差: {np.max(pos_err):.3f} mm")
    print(f"  位置误差标准差: {np.std(pos_err):.3f} mm")


    return {
        'abs_err_x': abs_err_x,
        'abs_err_y': abs_err_y,
        'abs_err_z': abs_err_z,
        'pos_err': pos_err,
        'name': model_name
    }

errors_physics = calculate_model_errors(real_coords, physics_sim_coords, "纯物理仿真")
errors_nn_corrected = calculate_model_errors(real_coords, nn_corrected_coords, "神经网络修正后")

# --- 5. 绘制对比误差箱线图 ---
print("\n--- 正在绘制对比误差箱线图 ---")
plt.figure(figsize=(12, 7)) # 可以根据需要调整图形大小

# 定义要绘制的误差类型和对应的标签
error_types_data_physics = [
    errors_physics['abs_err_x'],
    errors_physics['abs_err_y'],
    errors_physics['abs_err_z'],
    errors_physics['pos_err']
]
error_types_data_nn = [
    errors_nn_corrected['abs_err_x'],
    errors_nn_corrected['abs_err_y'],
    errors_nn_corrected['abs_err_z'],
    errors_nn_corrected['pos_err']
]
labels_error_types = ['X 轴绝对误差', 'Y 轴绝对误差', 'Z 轴绝对误差', '三维位置误差']

n_groups = len(labels_error_types)
bar_width = 0.35  # 每个箱子的宽度
index = np.arange(n_groups) # 每组误差类型的中心位置

# 自定义箱线图颜色和样式，使其更接近你提供的 error.png
# 纯物理仿真的箱子样式 (例如，蓝色系)
boxprops_phys = dict(facecolor='skyblue', color='black', linewidth=1.2, alpha=0.8)
medianprops = dict(color='navy', linewidth=1.5) # 中位数线深色
whiskerprops = dict(color='black', linewidth=1.2, linestyle='--')
capprops = dict(color='black', linewidth=1.2)

# 神经网络修正后的箱子样式 (例如，橙色/红色系)
boxprops_nn = dict(facecolor='lightcoral', color='black', linewidth=1.2, alpha=0.8)
# medianprops, whiskerprops, capprops 可以共用，或为NN模型也单独定义

# 绘制纯物理仿真的箱线图
# positions 参数用于指定每个箱子的中心点在x轴上的位置
bp1 = plt.boxplot(error_types_data_physics,
                  positions=index - bar_width / 2, # 将箱子向左移动半个宽度
                  widths=bar_width,
                  patch_artist=True, # 允许填充颜色
                  boxprops=boxprops_phys,
                  medianprops=medianprops,
                  whiskerprops=whiskerprops,
                  capprops=capprops,
                  showfliers=False, # 根据需要决定是否显示异常点
                  manage_ticks=False) # 我们将手动设置刻度

# 绘制神经网络修正后的箱线图
bp2 = plt.boxplot(error_types_data_nn,
                  positions=index + bar_width / 2, # 将箱子向右移动半个宽度
                  widths=bar_width,
                  patch_artist=True,
                  boxprops=boxprops_nn,
                  medianprops=medianprops, # 可以为NN模型的中位数线用不同颜色，例如 'darkred'
                  whiskerprops=whiskerprops,
                  capprops=capprops,
                  showfliers=False,
                  manage_ticks=False)


plt.ylabel('误差值 (mm)', fontsize=14)
plt.title('模型误差对比：纯物理仿真 vs. 神经网络修正', fontsize=16, pad=20)
plt.xticks(index, labels_error_types, fontsize=12, rotation=10, ha="right") # 设置X轴刻度标签
plt.yticks(fontsize=11)
plt.grid(axis='y', linestyle=':', alpha=0.7)

# 添加图例
# 创建代理艺术家（proxy artists）用于图例
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, facecolor=boxprops_phys['facecolor'], edgecolor='black', label='纯物理仿真'),
    plt.Rectangle((0, 0), 1, 1, facecolor=boxprops_nn['facecolor'], edgecolor='black', label='神经网络修正后')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=11)

# 移除顶部和右侧的图表边框线，使其更像 error.png
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)


plt.tight_layout() # 自动调整子图参数，使之填充整个图像区域

# 保存图像
plt.savefig(OUTPUT_PLOT_PATH, dpi=300, bbox_inches='tight')
print(f"对比误差箱线图已保存至: {OUTPUT_PLOT_PATH}")
plt.show()

print("\n--- 绘图脚本执行完毕 ---")