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
# 例如，如果这个脚本和 main_new.py 在同一目录，且 spy_visualizer 在 visualizer 子目录
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir)) # 如果 visualizer 和 utils 与上一级目录相关

import utils # 从你的项目中导入
from spy_visualizer import ODE # 从你的项目中导入

# --- 中文显示设置 ---
try:
    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("[信息] 尝试设置字体为 'SimHei' 以支持中文。")
except Exception as e:
    print(f"[警告] 设置 'SimHei' 字体失败: {e}. Plots might not render Chinese characters correctly.")

def run_physical_simulation_and_collect_data(
    data_file_path,
    sheet_name,
    robot_params,
    pybullet_params):
    """
    运行物理仿真并收集真实数据和仿真数据。
    """
    print("--- 加载数据 ---")
    # absolute_lengths_mm_all: (N, 3) 输入的绝对绳长
    # real_xyz_mm_all: (N, 3) 真实的末端XYZ坐标
    # dl_sequence_m_all: (N, 3) 绳长变化量 (米)
    # L0_cables_mm_initial: (3,) 初始绳长 (毫米)
    absolute_lengths_mm_all, real_xyz_mm_all, dl_sequence_m_all, L0_cables_mm_initial = \
        utils.load_and_preprocess_data(data_file_path, sheet_name)

    print("--- 初始化 PyBullet (DIRECT mode for speed) ---")
    if p.isConnected():
        p.disconnect()
    physicsClientId = p.connect(p.DIRECT) # 使用DIRECT模式，不显示GUI，加快数据生成
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(pybullet_params['simulationStepTime'])
    # planeId = p.loadURDF("plane.urdf") # 在DIRECT模式下通常不需要

    print("--- 初始化 ODE 对象 ---")
    my_ode = ODE(initial_length_m=robot_params['initial_length'],
                 cable_distance_m=robot_params['cable_distance'],
                 axial_coupling_coefficient=robot_params['axial_strain_coefficient'])

    # --- 设置基座 ---
    base_pos = np.array(pybullet_params['base_pos'])
    base_ori_euler = np.array(pybullet_params['base_ori_euler'])
    base_ori = p.getQuaternionFromEuler(base_ori_euler)

    simulated_xyz_mm_raw_all = [] # 只收集原始物理模型的仿真结果

    num_data_rows = len(dl_sequence_m_all)
    print(f"开始仿真 {num_data_rows} 个数据点...")

    for i in range(num_data_rows):
        if (i+1) % 500 == 0:
            print(f"  处理到第 {i+1}/{num_data_rows} 个数据点...")

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
            # 提取末端位置 (与 main_new.py 和 parameter_optimizer.py 中一致)
            pos_tip_local_sim = np.array([sol[0, -1], sol[2, -1], sol[1, -1]])
            pos_tip_world_tuple_sim, _ = p.multiplyTransforms(base_pos, base_ori, pos_tip_local_sim, [0,0,0,1])
            pos_tip_world_m_sim = np.array(pos_tip_world_tuple_sim)

        # 转换到毫米并应用utils.py中的坐标变换约定
        # 注意：这里的 Z 轴偏移 (如 -480) 是模型的一部分，需要在这里应用，
        # 因为我们比较的是最终的 sim_X/Y/Z_mm
        sim_x_mm = pos_tip_world_m_sim[0] * 1000.0 if not np.isnan(pos_tip_world_m_sim[0]) else np.nan
        sim_y_mm = pos_tip_world_m_sim[1] * 1000.0 if not np.isnan(pos_tip_world_m_sim[1]) else np.nan
        # 在 utils.append_result 中，sim_z_mm = sim_xyz_m[2] * 1000.0 - 480
        # 所以，如果 robot_params 中定义了 z_offset_for_comparison，我们在这里应用它
        sim_z_mm = (pos_tip_world_m_sim[2] * 1000.0 - robot_params.get('z_offset_for_comparison_mm', 0.0)
                    if not np.isnan(pos_tip_world_m_sim[2]) else np.nan)
        
        simulated_xyz_mm_raw_all.append([sim_x_mm, sim_y_mm, sim_z_mm])

    print("仿真完成。")
    p.disconnect()

    return real_xyz_mm_all, np.array(simulated_xyz_mm_raw_all)


def plot_comparison_and_errors(real_xyz, sim_xyz, title_suffix="物理模型"):
    """
    绘制真实轨迹、仿真轨迹以及它们之间的误差。
    """
    # 移除包含NaN的行以进行公平比较和误差计算
    valid_mask_real = ~np.isnan(real_xyz).any(axis=1)
    valid_mask_sim = ~np.isnan(sim_xyz).any(axis=1)
    valid_mask = valid_mask_real & valid_mask_sim
    
    real_xyz_valid = real_xyz[valid_mask]
    sim_xyz_valid = sim_xyz[valid_mask]
    
    if len(real_xyz_valid) == 0:
        print("没有有效的仿真或真实数据点可供比较。")
        return

    print(f"用于绘图和计算误差的有效数据点数量: {len(real_xyz_valid)}")

    # 计算误差
    errors = real_xyz_valid - sim_xyz_valid # error = real - sim
    error_x = errors[:, 0]
    error_y = errors[:, 1]
    error_z = errors[:, 2]
    error_3d = np.linalg.norm(errors, axis=1)

    # 计算统计指标
    mae_x = mean_absolute_error(real_xyz_valid[:, 0], sim_xyz_valid[:, 0])
    mae_y = mean_absolute_error(real_xyz_valid[:, 1], sim_xyz_valid[:, 1])
    mae_z = mean_absolute_error(real_xyz_valid[:, 2], sim_xyz_valid[:, 2])
    mae_3d = np.mean(error_3d)
    
    rmse_x = np.sqrt(mean_squared_error(real_xyz_valid[:, 0], sim_xyz_valid[:, 0]))
    rmse_y = np.sqrt(mean_squared_error(real_xyz_valid[:, 1], sim_xyz_valid[:, 1]))
    rmse_z = np.sqrt(mean_squared_error(real_xyz_valid[:, 2], sim_xyz_valid[:, 2]))
    rmse_3d = np.sqrt(np.mean(error_3d**2))

    print(f"\n--- {title_suffix} 性能评估 (单位: mm) ---")
    print(f"MAE X: {mae_x:.3f}, Y: {mae_y:.3f}, Z: {mae_z:.3f}")
    print(f"平均 3D MAE: {mae_3d:.3f}")
    print(f"RMSE X: {rmse_x:.3f}, Y: {rmse_y:.3f}, Z: {rmse_z:.3f}")
    print(f"平均 3D RMSE: {rmse_3d:.3f}")

    plot_indices = np.arange(len(real_xyz_valid))

    # --- 绘制轨迹对比图 ---
    fig_traj, axs_traj = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig_traj.suptitle(f'真实轨迹 vs {title_suffix}仿真轨迹 (3D MAE={mae_3d:.3f} mm)', fontsize=16)

    # X 轴
    axs_traj[0].plot(plot_indices, real_xyz_valid[:, 0], label='真实 X (mm)', color='darkorange', linewidth=1.5)
    axs_traj[0].plot(plot_indices, sim_xyz_valid[:, 0], label=f'{title_suffix} X (MAE={mae_x:.2f}mm)', color='dodgerblue', linestyle='--', linewidth=1.2)
    axs_traj[0].set_ylabel('X 坐标 (mm)')
    axs_traj[0].set_title('X 轴坐标对比')
    axs_traj[0].legend()
    axs_traj[0].grid(True, linestyle=':')

    # Y 轴
    axs_traj[1].plot(plot_indices, real_xyz_valid[:, 1], label='真实 Y (mm)', color='darkorange', linewidth=1.5)
    axs_traj[1].plot(plot_indices, sim_xyz_valid[:, 1], label=f'{title_suffix} Y (MAE={mae_y:.2f}mm)', color='dodgerblue', linestyle='--', linewidth=1.2)
    axs_traj[1].set_ylabel('Y 坐标 (mm)')
    axs_traj[1].set_title('Y 轴坐标对比')
    axs_traj[1].legend()
    axs_traj[1].grid(True, linestyle=':')

    # Z 轴
    axs_traj[2].plot(plot_indices, real_xyz_valid[:, 2], label='真实 Z (mm)', color='darkorange', linewidth=1.5)
    axs_traj[2].plot(plot_indices, sim_xyz_valid[:, 2], label=f'{title_suffix} Z (MAE={mae_z:.2f}mm)', color='dodgerblue', linestyle='--', linewidth=1.2)
    axs_traj[2].set_ylabel('Z 坐标 (mm)')
    axs_traj[2].set_title('Z 轴坐标对比')
    axs_traj[2].legend()
    axs_traj[2].grid(True, linestyle=':')
    axs_traj[2].set_xlabel('样本序号 (有效数据点)')
    fig_traj.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

    # --- 绘制误差曲线图 ---
    fig_err, axs_err = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig_err.suptitle(f'{title_suffix}仿真误差 (真实 - 仿真)', fontsize=16)

    axs_err[0].plot(plot_indices, error_x, label=f'误差 X (MAE={mae_x:.2f}mm)', color='orangered')
    axs_err[0].set_ylabel('误差 X (mm)')
    axs_err[0].set_title('X 轴误差')
    axs_err[0].legend()
    axs_err[0].grid(True, linestyle=':')

    axs_err[1].plot(plot_indices, error_y, label=f'误差 Y (MAE={mae_y:.2f}mm)', color='limegreen')
    axs_err[1].set_ylabel('误差 Y (mm)')
    axs_err[1].set_title('Y 轴误差')
    axs_err[1].legend()
    axs_err[1].grid(True, linestyle=':')

    axs_err[2].plot(plot_indices, error_z, label=f'误差 Z (MAE={mae_z:.2f}mm)', color='mediumpurple')
    axs_err[2].set_ylabel('误差 Z (mm)')
    axs_err[2].set_title('Z 轴误差')
    axs_err[2].legend()
    axs_err[2].grid(True, linestyle=':')
    
    axs_err[3].plot(plot_indices, error_3d, label=f'3D 欧氏误差 (平均={mae_3d:.2f}mm)', color='deepskyblue')
    axs_err[3].set_ylabel('3D 误差 (mm)')
    axs_err[3].set_title('3D 欧氏距离误差')
    axs_err[3].legend()
    axs_err[3].grid(True, linestyle=':')
    axs_err[3].set_xlabel('样本序号 (有效数据点)')

    fig_err.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    # --- 定义输入文件和参数 ---
    # 使用你在 main_new.py 或参数辨识后确定的参数
    DATA_FILE_PATH = 'D:/data/load_data/random_data_clean.xlsx' # 确保路径正确
    SHEET_NAME = 'Sheet1'

    # 这里的参数应该与你想要评估的物理模型版本一致
    # 例如，这是你 main_new.py 中的初始参数
    robot_physical_params = {
        'cable_distance': 0.035,
        'initial_length': 0.12,
        'number_of_segment': 1, # L0_seg 会据此计算
        'axial_strain_coefficient': 0, # k_strain in ODE
        'AXIAL_ACTION_SCALE': 0.77,
        'z_offset_for_comparison_mm': 480.0 # 这是你 utils.py 中 sim_z_mm 计算时减去的那个值
    }
    robot_physical_params['L0_seg'] = robot_physical_params['initial_length'] / robot_physical_params['number_of_segment']

    # PyBullet仿真环境参数 (从main_new.py中提取)
    pybullet_sim_params = {
        'simulationStepTime': 0.0001,
        'base_pos': [0, 0, 0.6],
        'base_ori_euler': [-math.pi / 2.0, 0, math.pi / 4.7]
    }

    # --- 运行仿真并收集数据 ---
    real_data_xyz_mm, sim_data_xyz_mm_raw = run_physical_simulation_and_collect_data(
        DATA_FILE_PATH,
        SHEET_NAME,
        robot_physical_params,
        pybullet_sim_params
    )

    # --- 绘图比较 ---
    plot_comparison_and_errors(real_data_xyz_mm, sim_data_xyz_mm_raw, title_suffix="物理模型")

    # 如果你做了参数辨识，并且想对比辨识后的参数效果：
    # 1. 定义一组新的 robot_physical_params_identified
    # robot_physical_params_identified = {
    #     'cable_distance': identified_val_1,
    #     'initial_length': identified_val_2,
    #      # ... 其他辨识出的参数
    #     'z_offset_for_comparison_mm': identified_z_offset
    # }
    # robot_physical_params_identified['L0_seg'] = robot_physical_params_identified['initial_length'] / robot_physical_params_identified['number_of_segment']
    #
    # 2. 再次调用 run_physical_simulation_and_collect_data
    # _, sim_data_xyz_mm_identified = run_physical_simulation_and_collect_data(
    # DATA_FILE_PATH, SHEET_NAME, robot_physical_params_identified, pybullet_sim_params
    # )
    # 3. 再次调用 plot_comparison_and_errors
    # plot_comparison_and_errors(real_data_xyz_mm, sim_data_xyz_mm_identified, title_suffix="辨识后物理模型")