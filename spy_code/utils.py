import math
import os
import pandas as pd
import numpy as np
import pybullet as p

def load_and_preprocess_data(file_path, sheet_name):
    print(f"[信息] 正在加载数据文件: {file_path}")
    
    df_input = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
    print(f"[成功] 已加载数据，共 {len(df_input)} 行。")

    absolute_lengths_mm = df_input[['cblen1', 'cblen2', 'cblen3']].values
    real_xyz_mm = df_input[['X', 'Y', 'Z']].values
    print(f"[信息] 成功提取 {len(absolute_lengths_mm)} 行绳长和真实 XYZ 坐标。")

    L0_cables_mm = absolute_lengths_mm[0]
    print(f"[假设] 使用第一行 L0(mm): {L0_cables_mm}")
    absolute_lengths_m = absolute_lengths_mm / 1000.0
    L0_cables_m = L0_cables_mm / 1000.0
    dl_sequence_m = absolute_lengths_m - L0_cables_m
    print(f"[信息] 已计算 {len(dl_sequence_m)} 行 dl (m)。")

    return absolute_lengths_mm, real_xyz_mm, dl_sequence_m, L0_cables_mm

def initialize_results_storage():
    return {
        'cblen1_mm': [], 'cblen2_mm': [], 'cblen3_mm': [],
        'X_real_mm': [], 'Y_real_mm': [], 'Z_real_mm': [],
        'sim_X_mm': [], 'sim_Y_mm': [], 'sim_Z_mm': []
    }

def append_result(results_dict, cblen_mm, real_xyz_mm, sim_xyz_m):
    """
    将单步的输入和仿真结果追加到结果字典中。

    Args:
        results_dict (dict): 要追加到的结果字典。
        cblen_mm (np.ndarray): 当前步的输入绳长 (1x3), 单位 mm。
        real_xyz_mm (np.ndarray): 当前步的真实坐标 (1x3), 单位 mm。
        sim_xyz_m (np.ndarray): 当前步的仿真末端坐标 (1x3), 单位 m。
    """
    results_dict['cblen1_mm'].append(cblen_mm[0])
    results_dict['cblen2_mm'].append(cblen_mm[1])
    results_dict['cblen3_mm'].append(cblen_mm[2])
    results_dict['X_real_mm'].append(real_xyz_mm[0])
    results_dict['Y_real_mm'].append(real_xyz_mm[1])
    results_dict['Z_real_mm'].append(real_xyz_mm[2])

    sim_x_mm = sim_xyz_m[0] * 1000.0 if not np.isnan(sim_xyz_m[0]) else np.nan
    sim_y_mm = sim_xyz_m[1] * -1000.0 if not np.isnan(sim_xyz_m[1]) else np.nan
    sim_z_mm = sim_xyz_m[2] * 1000.0 - 480 if not np.isnan(sim_xyz_m[2]) else np.nan
    results_dict['sim_X_mm'].append(sim_x_mm)
    results_dict['sim_Y_mm'].append(sim_y_mm)
    results_dict['sim_Z_mm'].append(sim_z_mm)

def save_results_to_excel(results_dict, output_path):
    """
    将结果字典保存到 Excel 文件 (无错误检查简化版)。

    Args:
        results_dict (dict): 包含所有仿真结果的字典。
        output_path (str): 输出 Excel 文件的路径。
    """
    print("\n--- 保存仿真结果 ---")
    output_column_order = [
        'cblen1_mm', 'cblen2_mm', 'cblen3_mm',
        'X_real_mm', 'Y_real_mm', 'Z_real_mm',
        'sim_X_mm', 'sim_Y_mm', 'sim_Z_mm'
    ]
    output_results_df = pd.DataFrame(results_data)
    output_results_df = output_results_df[output_column_order]

    print(f"[信息] 正在将 {len(output_results_df)} 条结果 ({len(output_column_order)} 列) 保存到: {output_path}")
    output_results_df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"[成功] 结果已保存至: {os.path.abspath(output_path)}")

def calculate_orientation(point1, point2):
    """根据两点计算方向 (四元数)"""
    diff = np.array(point2) - np.array(point1)
    norm_diff = np.linalg.norm(diff)
    # 保留对重合点的检查以避免数学错误
    if norm_diff < 1e-6:
        return p.getQuaternionFromEuler([0, 0, 0]), [0, 0, 0]
    # 保留对垂直情况的检查以计算 Yaw
    if np.linalg.norm(diff[:2]) < 1e-6:
        yaw = 0
    else:
        yaw = math.atan2(diff[1], diff[0])
    pitch = math.atan2(-diff[2], math.sqrt(diff[0]**2 + diff[1]**2))
    roll = 0
    return p.getQuaternionFromEuler([roll, pitch, yaw]), [roll, pitch, yaw]

def calculate_curvatures_from_dl(dl_segment, d, L0_seg, num_cables=3):
    """根据绳长变化量计算曲率 ux, uy (假设输入有效且为3绳索)。"""
    ux = 0.0
    uy = 0.0
    # 保留对半径 d 的检查以避免除零
    if abs(d) < 1e-9:
        print("警告: 绳索半径 d 接近于零。")
        return ux, uy
    # 移除对 dl_segment 长度的检查
    # 移除对 num_cables == 3 的检查，直接按 3 绳索处理
    dl1 = dl_segment[0] # 如果 dl_segment 长度不足会在此处中断
    dl2 = dl_segment[1]
    dl3 = dl_segment[2]
    uy = -dl1 / d
    denominator_ux = d * math.sqrt(3.0)
    # 保留对 ux 分母的检查以避免除零
    if abs(denominator_ux) > 1e-9:
        ux = (dl3 - dl2) / denominator_ux
    else:
        ux = 0.0 # 分母为零时 ux 定义为 0
        print("警告: 计算 ux 时分母接近零。")
    return ux, uy


    """
    使用针对 3 根对称缆绳的标准公式，根据不同的缆绳长度变化量计算曲率 ux, uy。

    假设缆绳 1 位于 0 度，缆绳 2 位于 120 度，缆绳 3 位于 240 度，
    相对于与 uy 弯曲相关的轴。

    Args:
        dl_segment (np.ndarray): 包含缆绳长度变化量 [dl1, dl2, dl3] 的数组 (单位：米)。
        d (float): 缆绳到中心骨干的径向距离 (单位：米)。
        L0_seg (float): 段的初始长度 (单位：米)。用于比例缩放。

    Returns:
        tuple: (ux, uy) 曲率 (单位：1/米)。
               ux: 绕局部 y 轴的曲率。
               uy: 绕局部 x 轴的曲率。
    """
    ux = 0.0
    uy = 0.0

    # 对有效输入进行基本检查
    if abs(d) < 1e-9:
        print("[警告] calculate_curvatures_from_dl_v2: 缆绳距离 'd' 接近于零。")
        return ux, uy
    if abs(L0_seg) < 1e-9:
        print("[警告] calculate_curvatures_from_dl_v2: 段长度 'L0_seg' 接近于零。")
        return ux, uy

    dl1 = dl_segment[0]
    dl2 = dl_segment[1]
    dl3 = dl_segment[2]

    # 分母部分 L*d
    # 我们使用 L0_seg 作为参考长度 L 来计算曲率贡献
    Ld = L0_seg * d

    # 计算 ux (绕 y 轴弯曲)
    ux_denominator = Ld * math.sqrt(3.0)
    if abs(ux_denominator) > 1e-9:
        # 注意: 原始代码是 dl3 - dl2。根据缆绳 2/3 相对于 y 轴的定义，
        # 可能是 dl2 - dl3。我们暂时先保持 dl3 - dl2。
        ux = (dl3 - dl2) / ux_denominator
    else:
        ux = 0.0

    # 计算 uy (绕 x 轴弯曲)
    uy_denominator = 3.0 * Ld
    if abs(uy_denominator) > 1e-9:
         uy = (2.0 * dl1 - dl2 - dl3) / uy_denominator
    else:
         uy = 0.0

    return ux, uy

def calculate_curvatures_from_dl_v3(dl_segment, d, L0_seg, AXIAL_ACTION_SCALE=1.0):
    """
    计算曲率 ux, uy，使用标准公式，但使用当前估计长度而非初始长度进行缩放。

    Args:
        dl_segment (np.ndarray): 包含缆绳长度变化量 [dl1, dl2, dl3] 的数组 (单位：米)。
        d (float): 缆绳到中心骨干的径向距离 (单位：米)。
        L0_seg (float): 段的初始长度 (单位：米)。
        axial_scale (float): 应用于 avg_dl 以估计长度变化的比例因子。默认为 1.0。

    Returns:
        tuple: (ux, uy) 曲率 (单位：1/米)。
               ux: 绕局部 y 轴的曲率。
               uy: 绕局部 x 轴的曲率。
    """
    ux = 0.0
    uy = 0.0


    dl1 = dl_segment[0]
    dl2 = dl_segment[1]
    dl3 = dl_segment[2]

    # 根据平均 dl 和比例因子估计当前长度 L
    avg_dl = (dl1 + dl2 + dl3) / 3.0
    L_current_estimate = L0_seg + avg_dl * AXIAL_ACTION_SCALE

    # 使用当前估计长度计算分母部分 L*d
    Ld = L_current_estimate * d

    # 计算 ux (绕 y 轴弯曲)
    ux_denominator = Ld * math.sqrt(3.0)
    if abs(ux_denominator) > 1e-9:
        ux = (dl3 - dl2) / ux_denominator
    else:
        ux = 0.0

    # 计算 uy (绕 x 轴弯曲)
    uy_denominator = 3.0 * Ld
    if abs(uy_denominator) > 1e-9:
         uy = (2.0 * dl1 - dl2 - dl3) / uy_denominator
    else:
         uy = 0.0

    return ux, uy
