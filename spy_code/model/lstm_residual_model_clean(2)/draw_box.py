import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # 可选，用于更美观的图形样式
import os

# --- 中文显示和负号显示设置 ---
try:
    plt.rcParams['font.family'] = ['SimHei'] # 或者 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False
    print("[信息] 尝试设置中文字体以支持中文显示。")
except Exception as e:
    print(f"[警告] 设置中文字体失败: {e}. 图中中文可能无法正常显示。")

# --- 配置 ---
EXCEL_FILE_PATH = 'D:/data/save_data/lstm_test_results_output.xlsx' # 修改为你的Excel文件路径
OUTPUT_PLOT_PATH = 'D:/data/save_model/lstm_residual_model_clean(2)/error_boxplot_comparison.png' # 修改为你希望保存图片的路径

# 检查文件是否存在
if not os.path.exists(EXCEL_FILE_PATH):
    print(f"[错误] Excel文件未找到: {EXCEL_FILE_PATH}")
    exit()

# --- 1. 加载数据 ---
try:
    df_results = pd.read_excel(EXCEL_FILE_PATH, engine='openpyxl')
    print(f"数据加载成功，共 {len(df_results)} 行。")
except Exception as e:
    print(f"加载Excel文件时出错: {e}")
    exit()

# --- 2. 计算或提取误差 ---
# 确保所有需要的列都存在
required_cols_for_plot = {
    'real_x': 'X_real_mm', 'real_y': 'Y_real_mm', 'real_z': 'Z_real_mm',
    'sim_raw_x': 'sim_X_raw_mm', 'sim_raw_y': 'sim_Y_raw_mm', 'sim_raw_z': 'sim_Z_raw_mm',
    'sim_corrected_x': 'X_corrected_mm', 'sim_corrected_y': 'Y_corrected_mm', 'sim_corrected_z': 'Z_corrected_mm'
}

# 检查列是否存在，如果不存在则尝试从其他列计算（例如，如果Excel中直接有Error列）
missing_cols = []
for internal_name, excel_col_name in required_cols_for_plot.items():
    if excel_col_name not in df_results.columns:
        missing_cols.append(excel_col_name)

if missing_cols:
    print(f"[警告] Excel文件中缺少以下必需的列: {', '.join(missing_cols)}")
    print("请确保Excel文件包含真实坐标、原始仿真坐标和修正后仿真坐标的列。")
    # 如果有直接的Error列，也可以在这里添加逻辑来使用它们
    # 例如，如果 Error_X_mm 存在，就不需要从 real_x 和 sim_corrected_x 计算
    # 为简化起见，我们假设上面的列是必须的，并从中计算误差

    # 检查是否有直接的误差列，以防万一
    if 'Error_X_mm' in df_results.columns and 'Error_Y_mm' in df_results.columns and 'Error_Z_mm' in df_results.columns:
        print("检测到Excel文件中已存在 Error_X/Y/Z_mm 列，将优先使用它们作为修正后模型的误差。")
        # 此时，sim_corrected_x/y/z 可能不是必须的，但为了统一，我们还是要求它们存在
    else:
        if missing_cols: # 如果在有Error列的情况下，其他列仍缺失，则退出
             print("[错误] 即使有Error列，其他用于计算原始模型误差的列也缺失，无法继续。")
             exit()


# 计算原始物理模型的误差 (取绝对误差，因为箱线图通常关注误差的大小分布)
df_results['Error_Raw_X_abs'] = np.abs(df_results[required_cols_for_plot['real_x']] - df_results[required_cols_for_plot['sim_raw_x']])
df_results['Error_Raw_Y_abs'] = np.abs(df_results[required_cols_for_plot['real_y']] - df_results[required_cols_for_plot['sim_raw_y']])
df_results['Error_Raw_Z_abs'] = np.abs(df_results[required_cols_for_plot['real_z']] - df_results[required_cols_for_plot['sim_raw_z']])
df_results['Error_Raw_3D'] = np.linalg.norm(
    df_results[[required_cols_for_plot['real_x'], required_cols_for_plot['real_y'], required_cols_for_plot['real_z']]].values -
    df_results[[required_cols_for_plot['sim_raw_x'], required_cols_for_plot['sim_raw_y'], required_cols_for_plot['sim_raw_z']]].values,
    axis=1
)

# 计算/提取LSTM修正后模型的误差 (取绝对误差)
# 如果Excel中已经有Error_X_mm等列，它们可能是有符号的，这里我们取绝对值
if 'Error_X_mm' in df_results.columns: # 假设这些是 (real - corrected_sim)
    df_results['Error_Corrected_X_abs'] = np.abs(df_results['Error_X_mm'])
    df_results['Error_Corrected_Y_abs'] = np.abs(df_results['Error_Y_mm'])
    df_results['Error_Corrected_Z_abs'] = np.abs(df_results['Error_Z_mm'])
else:
    df_results['Error_Corrected_X_abs'] = np.abs(df_results[required_cols_for_plot['real_x']] - df_results[required_cols_for_plot['sim_corrected_x']])
    df_results['Error_Corrected_Y_abs'] = np.abs(df_results[required_cols_for_plot['real_y']] - df_results[required_cols_for_plot['sim_corrected_y']])
    df_results['Error_Corrected_Z_abs'] = np.abs(df_results[required_cols_for_plot['real_z']] - df_results[required_cols_for_plot['sim_corrected_z']])

if 'Error_3D_mm' in df_results.columns:
    df_results['Error_Corrected_3D'] = df_results['Error_3D_mm'] # 假设已经是欧氏距离误差
else:
    df_results['Error_Corrected_3D'] = np.linalg.norm(
        df_results[[required_cols_for_plot['real_x'], required_cols_for_plot['real_y'], required_cols_for_plot['real_z']]].values -
        df_results[[required_cols_for_plot['sim_corrected_x'], required_cols_for_plot['sim_corrected_y'], required_cols_for_plot['sim_corrected_z']]].values,
        axis=1
    )

# --- 3. 准备绘图数据 ---
# 我们将为每个轴（X, Y, Z）和3D误差分别绘制箱线图
# 每个图上包含两个箱子：一个是原始物理模型的，一个是LSTM修正后的
plot_data = [
    df_results['Error_Raw_X_abs'].dropna(), df_results['Error_Corrected_X_abs'].dropna(),
    df_results['Error_Raw_Y_abs'].dropna(), df_results['Error_Corrected_Y_abs'].dropna(),
    df_results['Error_Raw_Z_abs'].dropna(), df_results['Error_Corrected_Z_abs'].dropna(),
    df_results['Error_Raw_3D'].dropna(), df_results['Error_Corrected_3D'].dropna()
]

labels = [
    'X轴\n(原始物理模型)', 'X轴\n(LSTM修正后)',
    'Y轴\n(原始物理模型)', 'Y轴\n(LSTM修正后)',
    'Z轴\n(原始物理模型)', 'Z轴\n(LSTM修正后)',
    '3D误差\n(原始物理模型)', '3D误差\n(LSTM修正后)'
]

# 为了模仿你给的图片样式（两组颜色，每组内并列），我们需要调整一下数据结构和绘图方式
# 创建一个 "长格式" DataFrame 更适合 Seaborn 或 Matplotlib 的高级绘图
df_plot_long = pd.DataFrame({
    'Error_X_Raw': df_results['Error_Raw_X_abs'],
    'Error_X_Corrected': df_results['Error_Corrected_X_abs'],
    'Error_Y_Raw': df_results['Error_Raw_Y_abs'],
    'Error_Y_Corrected': df_results['Error_Corrected_Y_abs'],
    'Error_Z_Raw': df_results['Error_Raw_Z_abs'],
    'Error_Z_Corrected': df_results['Error_Corrected_Z_abs'],
    'Error_3D_Raw': df_results['Error_Raw_3D'],
    'Error_3D_Corrected': df_results['Error_Corrected_3D'],
})

# --- 4. 绘制箱线图 ---
# 使用Seaborn可以更容易实现分组和美化，但这里先用Matplotlib基础绘图，并尝试模拟分组颜色
try:
    # 尝试使用Seaborn的推荐方法设置主题
    sns.set_theme(style="whitegrid", font="SimHei" if 'SimHei' in plt.rcParams['font.family'] else None)
    # 如果只想模仿旧的样式，可以尝试：
    # sns.set_style("whitegrid")
    print("[信息] 已使用 Seaborn set_theme(style='whitegrid') 设置绘图样式。")
except Exception as e_seaborn:
    print(f"[警告] 使用 sns.set_theme 失败: {e_seaborn}. 将尝试Matplotlib的 'seaborn-v0_8-whitegrid' 或 'ggplot' 样式。")
    try:
        plt.style.use('seaborn-v0_8-whitegrid') # 尝试一个可能的变体
    except:
        try:
            plt.style.use('ggplot') # 一个常用的备选样式
            print("[信息] 已使用备选样式 'ggplot'。")
        except Exception as e_style:
            print(f"[警告] 设置Matplotlib样式失败: {e_style}. 将使用默认样式。")


fig, ax = plt.subplots(figsize=(12, 7)) # 调整图像大小

# 定义颜色
color_raw = 'tab:blue'  # 原始模型的颜色 (类似你图中深蓝色)
color_corrected = 'tab:orange' # 修正后模型的颜色 (类似你图中橙色)

positions = np.array(range(len(plot_data))) + 1 # 箱子的位置
box_plot = ax.boxplot(plot_data,
                      labels=labels,
                      patch_artist=True, # 允许填充颜色
                      widths=0.6,
                      showfliers=True) # 是否显示异常值，可以设为False让图更干净

# 设置颜色
colors_for_boxes = [color_raw, color_corrected] * 4 # 重复颜色模式
for patch, color in zip(box_plot['boxes'], colors_for_boxes):
    patch.set_facecolor(color)
    patch.set_edgecolor('black') # 箱体边缘颜色

# 设置中位数线的颜色
for median in box_plot['medians']:
    median.set_color('black')
    median.set_linewidth(1.5)

# 设置whiskers, caps, fliers的颜色 (可选，可以让它们与箱体颜色匹配或统一为黑色)
for i in range(len(plot_data)):
    box_plot['whiskers'][i*2].set_color(colors_for_boxes[i]) # 上须
    box_plot['whiskers'][i*2 + 1].set_color(colors_for_boxes[i]) # 下须
    box_plot['caps'][i*2].set_color(colors_for_boxes[i]) # 上帽
    box_plot['caps'][i*2 + 1].set_color(colors_for_boxes[i]) # 下帽
    # box_plot['fliers'][i].set(markerfacecolor=colors_for_boxes[i], markeredgecolor=colors_for_boxes[i], alpha=0.5)


ax.set_title('不同模型误差对比箱线图', fontsize=16)
ax.set_ylabel('绝对误差 (mm)', fontsize=14)
ax.set_xlabel('误差类型与模型', fontsize=14)
plt.xticks(rotation=15, ha='right', fontsize=10) # 旋转标签防止重叠
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.7)
ax.set_ylim(bottom=0) # 误差通常从0开始

# 添加图例说明颜色 (如果需要，但箱线图通常通过标签说明)
# import matplotlib.patches as mpatches
# legend_patches = [mpatches.Patch(color=color_raw, label='原始物理模型'),
#                   mpatches.Patch(color=color_corrected, label='LSTM修正后模型')]
# ax.legend(handles=legend_patches, loc='upper right')

# 模仿你图片中的简洁风格 (移除顶部和右侧的spines)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=5)


plt.tight_layout() # 自动调整布局

# 保存图像
try:
    plt.savefig(OUTPUT_PLOT_PATH, dpi=300, bbox_inches='tight')
    print(f"箱线图已保存到: {OUTPUT_PLOT_PATH}")
except Exception as e:
    print(f"保存图像失败: {e}")

plt.show()

# --- 打印一些统计数据，方便对比 ---
print("\n--- 误差统计摘要 (中位数) ---")
for i in range(0, len(plot_data), 2):
    median_raw = np.median(plot_data[i])
    median_corrected = np.median(plot_data[i+1])
    print(f"{labels[i].splitlines()[0]}:")
    print(f"  原始模型中位误差: {median_raw:.3f} mm")
    print(f"  修正模型中位误差: {median_corrected:.3f} mm")

print("\n--- 误差统计摘要 (平均值 MAE) ---")
for i in range(0, len(plot_data), 2):
    mean_raw = np.mean(plot_data[i])
    mean_corrected = np.mean(plot_data[i+1])
    print(f"{labels[i].splitlines()[0]}:")
    print(f"  原始模型平均绝对误差: {mean_raw:.3f} mm")
    print(f"  修正模型平均绝对误差: {mean_corrected:.3f} mm")