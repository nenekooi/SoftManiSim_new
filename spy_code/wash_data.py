import sys
import os
import numpy as np
import pandas as pd
import pybullet as p
import pybullet_data
import time
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 确保你的 utils.py 和 spy_visualizer.py 可以在 Python路径中被找到
current_dir = os.path.dirname(os.path.abspath(__file__))
# 你可能需要根据你的项目结构调整这里的路径
# 例如，如果 visualizer 和 utils 在上一级目录的某个子文件夹中
# sys.path.append(os.path.abspath(os.path.join(current_dir, '..'))) # 示例

import utils # 从你的项目中导入
from spy_visualizer import ODE # 从你的项目中导入

# --- 中文显示设置 ---
try:
    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("[信息] 尝试设置字体为 'SimHei' 以支持中文。")
except Exception as e:
    print(f"[警告] 设置 'SimHei' 字体失败: {e}. Plots might not render Chinese characters correctly.")

def run_simulation_for_data_cleaning(
    data_file_path,
    sheet_name,
    robot_params,
    pybullet_params):
    """
    运行物理仿真并返回真实数据、仿真数据以及原始的DataFrame索引。
    这个函数与之前的 run_physical_simulation_and_collect_data 类似，
    但会返回原始DataFrame的索引，方便后续筛选。
    """
    print("--- 加载原始数据进行清理 ---")
    # df_input 包含了所有原始列
    df_input = pd.read_excel(data_file_path, sheet_name=sheet_name, engine='openpyxl')
    
    absolute_lengths_mm_all = df_input[['cblen1', 'cblen2', 'cblen3']].values
    real_xyz_mm_all = df_input[['X', 'Y', 'Z']].values
    
    # 从 utils.load_and_preprocess_data 获取 dl_sequence_m 的逻辑
    L0_cables_mm_initial = absolute_lengths_mm_all[0]
    absolute_lengths_m = absolute_lengths_mm_all / 1000.0
    L0_cables_m = L0_cables_mm_initial / 1000.0
    dl_sequence_m_all = absolute_lengths_m - L0_cables_m

    print("--- 初始化 PyBullet (DIRECT mode for speed) ---")
    if p.isConnected():
        try:
            p.disconnect()
        except p.error as e:
            print(f"PyBullet disconnect error (ignoring): {e}")

    physicsClientId = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(pybullet_params['simulationStepTime'])

    print("--- 初始化 ODE 对象 ---")
    my_ode = ODE(initial_length_m=robot_params['initial_length'],
                 cable_distance_m=robot_params['cable_distance'],
                 axial_coupling_coefficient=robot_params['axial_strain_coefficient'])

    base_pos = np.array(pybullet_params['base_pos'])
    base_ori_euler = np.array(pybullet_params['base_ori_euler'])
    base_ori = p.getQuaternionFromEuler(base_ori_euler)

    simulated_xyz_mm_raw_all = []

    num_data_rows = len(dl_sequence_m_all)
    print(f"开始仿真 {num_data_rows} 个数据点以计算误差...")

    for i in range(num_data_rows):
        if (i + 1) % 1000 == 0:
            print(f"  处理到第 {i + 1}/{num_data_rows} 个数据点...")

        dl_segment = dl_sequence_m_all[i]
        my_ode._reset_y0()
        ux, uy = utils.calculate_curvatures_from_dl_v4(
            dl_segment,
            robot_params['cable_distance'],
            robot_params['L0_seg'],
            robot_params['AXIAL_ACTION_SCALE']
        )
        avg_dl = np.mean(dl_segment)
        commanded_length_change = avg_dl * robot_params['AXIAL_ACTION_SCALE']
        
        my_ode.set_kinematic_state_spy(commanded_length_change, ux, uy)
        sol = my_ode.odeStepFull()

        pos_tip_world_m_sim = np.array([np.nan, np.nan, np.nan])
        if sol is not None and sol.shape[1] >= 3:
            pos_tip_local_sim = np.array([sol[0, -1], sol[2, -1], sol[1, -1]])
            pos_tip_world_tuple_sim, _ = p.multiplyTransforms(base_pos, base_ori, pos_tip_local_sim, [0,0,0,1])
            pos_tip_world_m_sim = np.array(pos_tip_world_tuple_sim)

        sim_x_mm = pos_tip_world_m_sim[0] * 1000.0 if not np.isnan(pos_tip_world_m_sim[0]) else np.nan
        sim_y_mm = pos_tip_world_m_sim[1] * 1000.0 if not np.isnan(pos_tip_world_m_sim[1]) else np.nan
        sim_z_mm = (pos_tip_world_m_sim[2] * 1000.0 - robot_params.get('z_offset_for_comparison_mm', 0.0)
                    if not np.isnan(pos_tip_world_m_sim[2]) else np.nan)
        
        simulated_xyz_mm_raw_all.append([sim_x_mm, sim_y_mm, sim_z_mm])

    print("仿真完成。")
    p.disconnect()

    return df_input, real_xyz_mm_all, np.array(simulated_xyz_mm_raw_all)


def clean_data_based_on_simulation_error(
    df_original,
    real_xyz,
    sim_xyz,
    error_threshold_mm=None,
    percentile_threshold=None):
    """
    根据真实与仿真之间的3D误差筛选数据。
    可以指定一个固定的误差阈值，或者一个百分位阈值。

    Args:
        df_original (pd.DataFrame): 包含所有原始数据的DataFrame。
        real_xyz (np.ndarray): 真实的XYZ坐标 (N, 3)。
        sim_xyz (np.ndarray): 仿真的XYZ坐标 (N, 3)。
        error_threshold_mm (float, optional): 3D欧氏距离误差的绝对阈值。
                                              超过此阈值的数据点将被移除。
        percentile_threshold (float, optional): 要移除的数据的误差百分位数 (0-100)。
                                                例如，95 表示移除误差最大的5%的数据。
                                                如果同时指定 error_threshold_mm，则优先使用 error_threshold_mm。
    Returns:
        pd.DataFrame: 清理后的DataFrame。
        np.ndarray: 被保留的数据点的索引。
    """
    # 移除包含NaN的行以进行误差计算
    valid_mask_real = ~np.isnan(real_xyz).any(axis=1)
    valid_mask_sim = ~np.isnan(sim_xyz).any(axis=1)
    valid_mask_initial = valid_mask_real & valid_mask_sim
    
    real_xyz_valid = real_xyz[valid_mask_initial]
    sim_xyz_valid = sim_xyz[valid_mask_initial]
    
    if len(real_xyz_valid) == 0:
        print("警告：没有有效的仿真或真实数据点可供计算误差。返回原始数据。")
        return df_original, np.arange(len(df_original))

    # 计算每个有效数据点的3D欧氏距离误差
    errors_3d = np.linalg.norm(real_xyz_valid - sim_xyz_valid, axis=1)

    # 确定哪些行要保留
    if error_threshold_mm is not None:
        print(f"使用固定误差阈值: {error_threshold_mm} mm")
        keep_mask_for_valid_data = errors_3d <= error_threshold_mm
    elif percentile_threshold is not None:
        if not (0 < percentile_threshold < 100):
            raise ValueError("Percentile threshold must be between 0 and 100 (exclusive of 0 and 100 for practical use).")
        threshold_value = np.percentile(errors_3d, percentile_threshold)
        print(f"使用百分位阈值: {percentile_threshold}th percentile (误差值 <= {threshold_value:.3f} mm)")
        keep_mask_for_valid_data = errors_3d <= threshold_value
    else:
        print("警告：未指定误差阈值或百分位阈值。将不进行数据清理。")
        return df_original, np.arange(len(df_original))

    # 将 keep_mask_for_valid_data 映射回原始数据的索引
    # valid_indices_original 是 valid_mask_initial 为 True 的那些原始索引
    valid_indices_original = np.where(valid_mask_initial)[0]
    
    # kept_indices_in_valid_set 是在 errors_3d 中满足保留条件的点的索引
    kept_indices_in_valid_set = np.where(keep_mask_for_valid_data)[0]
    
    # final_kept_indices_original 是在 df_original 中最终要保留的行的索引
    final_kept_indices_original = valid_indices_original[kept_indices_in_valid_set]

    df_cleaned = df_original.iloc[final_kept_indices_original].copy()
    
    num_original = len(df_original)
    num_cleaned = len(df_cleaned)
    num_removed = num_original - num_cleaned
    
    print(f"原始数据点数量: {num_original}")
    print(f"清理后数据点数量: {num_cleaned} (移除了 {num_removed} 个点,占 {(num_removed/num_original)*100:.2f}%)")

    # 可选：绘制误差分布图以帮助选择阈值
    plt.figure(figsize=(10, 6))
    plt.hist(errors_3d, bins=100, alpha=0.7, label='3D 误差分布')
    if error_threshold_mm is not None:
        plt.axvline(error_threshold_mm, color='r', linestyle='--', label=f'固定阈值 ({error_threshold_mm:.2f}mm)')
    elif percentile_threshold is not None:
        plt.axvline(threshold_value, color='m', linestyle='--', label=f'{percentile_threshold}th 百分位阈值 ({threshold_value:.2f}mm)')
    plt.xlabel('3D 欧氏距离误差 (mm)')
    plt.ylabel('频数')
    plt.title('清理前仿真与真实数据的3D误差分布')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.show()

    return df_cleaned, final_kept_indices_original


if __name__ == "__main__":
    # --- 定义输入文件和参数 ---
    DATA_FILE_PATH = 'D:/data/load_data/random_data.xlsx' # 确保路径正确
    SHEET_NAME = 'Sheet1'
    CLEANED_DATA_OUTPUT_PATH = 'D:/data/load_data/random_data_clean.xlsx' # 清理后数据保存路径

    # 使用你认为最能代表机器人行为的物理模型参数
    # 这可能是你的初始参数，或者参数辨识后的参数
    robot_physical_params = {
        'cable_distance': 0.035,
        'initial_length': 0.12,
        'number_of_segment': 1,
        'axial_strain_coefficient': 0,
        'AXIAL_ACTION_SCALE': 0.771657,
        'z_offset_for_comparison_mm': 480.0 # 与仿真代码中 Z 轴调整一致
    }
    robot_physical_params['L0_seg'] = robot_physical_params['initial_length'] / robot_physical_params['number_of_segment']

    pybullet_sim_params = {
        'simulationStepTime': 0.0001,
        'base_pos': [0, 0, 0.6],
        'base_ori_euler': [-math.pi / 2.0, 0, math.pi / 4.7]
    }

    # --- 运行仿真以获取误差基准 ---
    df_original_data, real_data_xyz_mm, sim_data_xyz_mm_raw = run_simulation_for_data_cleaning(
        DATA_FILE_PATH,
        SHEET_NAME,
        robot_physical_params,
        pybullet_sim_params
    )

    # --- 设置清理参数 ---
    # 方式一：使用固定的3D误差阈值 (单位：毫米)
    # ERROR_THRESHOLD_MM = 20.0  # 例如，移除3D误差大于20mm的数据点
    # PERCENTILE_TO_KEEP = None # 设为None则使用固定阈值

    # 方式二：移除误差最大的 X% 的数据 (例如，移除误差最大的5%，即保留误差较小的95%)
    ERROR_THRESHOLD_MM = None
    PERCENTILE_THRESHOLD_FOR_REMOVAL = 95 # 保留误差在前95%的数据点，即移除误差最大的5%
                                        # 注意函数 clean_data_based_on_simulation_error 中的 percentile_threshold
                                        # 指的是误差的百分位数值，小于等于该误差值的数据被保留。

    # --- 执行数据清理 ---
    df_cleaned_data, _ = clean_data_based_on_simulation_error(
        df_original_data,
        real_data_xyz_mm,
        sim_data_xyz_mm_raw,
        error_threshold_mm=ERROR_THRESHOLD_MM,
        percentile_threshold=PERCENTILE_THRESHOLD_FOR_REMOVAL
    )

    # --- 保存清理后的数据 ---
    if not df_cleaned_data.empty:
        try:
            df_cleaned_data.to_excel(CLEANED_DATA_OUTPUT_PATH, index=False, engine='openpyxl')
            print(f"\n清理后的数据已保存到: {CLEANED_DATA_OUTPUT_PATH}")
        except Exception as e:
            print(f"保存清理后的Excel文件失败: {e}")
    else:
        print("没有数据被清理或保存，因为清理后数据集为空。")

    print("\n数据清理流程结束。")