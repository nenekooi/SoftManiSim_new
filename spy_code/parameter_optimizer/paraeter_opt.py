import numpy as np
import pandas as pd
from scipy.optimize import minimize
import pybullet as p
import pybullet_data
import math
import sys
import os
softmanisim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(softmanisim_path)
import utils
softmanisim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))
sys.path.append(softmanisim_path)
from visualizer.spy_visualizer import ODE

physicsClientId = -1 # 全局PyBullet客户端ID

def init_pybullet_direct():
    global physicsClientId
    if p.isConnected(physicsClientId): # 如果已经连接，先断开旧的
        try:
            p.disconnect(physicsClientId)
        except Exception as e:
            print(f"Warning: could not disconnect previous PyBullet instance: {e}")
    physicsClientId = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(0.0001)
    return physicsClientId

def run_simulation_for_params(params_dict_input, dl_sequence_m_all_data, L0_cables_mm_initial_data):
    global physicsClientId
    if not p.isConnected(physicsClientId) or physicsClientId == -1 :
        init_pybullet_direct()

    cable_distance_val = params_dict_input['cable_distance']
    initial_length_val = params_dict_input['initial_length']
    axial_strain_coefficient_val = params_dict_input['axial_strain_coefficient']
    AXIAL_ACTION_SCALE_val = params_dict_input['AXIAL_ACTION_SCALE']
    # z_offset_mm_val = params_dict_input['z_offset_mm'] # z_offset 在cost函数中处理

    L0_seg_val = initial_length_val / 1.0

    my_ode_instance = ODE(initial_length_m=initial_length_val,
                          cable_distance_m=cable_distance_val,
                          axial_coupling_coefficient=axial_strain_coefficient_val)

    base_pos_val = np.array([0, 0, 0.6])
    base_ori_euler_val = np.array([-math.pi / 2.0, 0, math.pi / 4.7])
    base_ori_val = p.getQuaternionFromEuler(base_ori_euler_val)

    simulated_xyz_mm_all_points = []
    num_data_rows_val = len(dl_sequence_m_all_data)

    for i in range(num_data_rows_val):
        dl_segment_val = dl_sequence_m_all_data[i]
        my_ode_instance._reset_y0()
        ux_val, uy_val = utils.calculate_curvatures_from_dl_v4(dl_segment_val,
                                                               cable_distance_val,
                                                               L0_seg_val,
                                                               AXIAL_ACTION_SCALE_val)
        avg_dl_val = np.mean(dl_segment_val)
        commanded_length_change_val = avg_dl_val * AXIAL_ACTION_SCALE_val
        
        my_ode_instance.set_kinematic_state_spy(commanded_length_change_val, ux_val, uy_val)
        sol_val = my_ode_instance.odeStepFull()

        pos_tip_world_m_sim_val = np.array([np.nan, np.nan, np.nan])
        if sol_val is not None and sol_val.shape[1] >= 3:
            pos_tip_local_sim_val = np.array([sol_val[0, -1], sol_val[2, -1], sol_val[1, -1]])
            pos_tip_world_tuple_sim_val, _ = p.multiplyTransforms(base_pos_val, base_ori_val, pos_tip_local_sim_val, [0,0,0,1])
            pos_tip_world_m_sim_val = np.array(pos_tip_world_tuple_sim_val)

        sim_x_mm_val = pos_tip_world_m_sim_val[0] * -1000.0 if not np.isnan(pos_tip_world_m_sim_val[0]) else np.nan
        sim_y_mm_val = pos_tip_world_m_sim_val[1] * -1000.0 if not np.isnan(pos_tip_world_m_sim_val[1]) else np.nan
        sim_z_mm_val = pos_tip_world_m_sim_val[2] * 1000.0 if not np.isnan(pos_tip_world_m_sim_val[2]) else np.nan
        simulated_xyz_mm_all_points.append([sim_x_mm_val, sim_y_mm_val, sim_z_mm_val])
    
    return np.array(simulated_xyz_mm_all_points)

def shutdown_pybullet():
    global physicsClientId
    if p.isConnected(physicsClientId) and physicsClientId !=-1:
        try:
            p.disconnect(physicsClientId)
        except Exception as e:
            print(f"Error disconnecting PyBullet: {e}")
        physicsClientId = -1
# --- 全局变量，用于存储加载的数据，避免重复加载 ---
_DATA_CACHE = {}

def load_data_once(data_file_path, sheet_name, num_samples=None):
    """加载数据，如果已加载则从缓存返回。可以选择加载部分样本。"""
    cache_key = (data_file_path, sheet_name, num_samples)
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]

    print(f"--- Loading data for optimization (max {num_samples} samples) ---")
    _, real_xyz_mm_all, dl_sequence_m_all, L0_cables_mm_initial = \
        utils.load_and_preprocess_data(data_file_path, sheet_name)

    if num_samples is not None and num_samples < len(real_xyz_mm_all):
        print(f"Using a subset of {num_samples} samples for optimization.")
        # 你可能想选择有代表性的样本，这里简单地取前N个
        indices = np.random.choice(len(real_xyz_mm_all), num_samples, replace=False)
        indices.sort() # 保持一定顺序性可能更好
        real_xyz_mm_subset = real_xyz_mm_all[indices]
        dl_sequence_m_subset = dl_sequence_m_all[indices]
    else:
        real_xyz_mm_subset = real_xyz_mm_all
        dl_sequence_m_subset = dl_sequence_m_all
    
    # L0_cables_mm_initial 仍然是基于完整数据集的第一行得到的，这是合理的
    _DATA_CACHE[cache_key] = (real_xyz_mm_subset, dl_sequence_m_subset, L0_cables_mm_initial)
    return real_xyz_mm_subset, dl_sequence_m_subset, L0_cables_mm_initial

# --- 代价函数 ---
# 参数顺序: cable_distance, initial_length, axial_strain_coefficient, AXIAL_ACTION_SCALE, z_offset_mm
PARAM_NAMES = ['cable_distance', 'initial_length', 'axial_strain_coefficient', 'AXIAL_ACTION_SCALE', 'z_offset_mm']

def cost_function(parameter_array, data_file, sheet, num_opt_samples):
    """
    代价函数，用于优化器。
    """
    params_dict = {name: val for name, val in zip(PARAM_NAMES, parameter_array)}
    
    # 加载数据子集
    real_xyz_mm, dl_sequence_m, L0_cables_mm = load_data_once(data_file, sheet, num_opt_samples)

    # 运行仿真
    simulated_xyz_mm = run_simulation_for_params(params_dict, dl_sequence_m, L0_cables_mm)

    # 处理潜在的NaN值 (如果仿真失败或产生NaN)
    # 在计算误差前，确保真实值和仿真值中对应的行都是有效的
    valid_mask = ~np.isnan(simulated_xyz_mm).any(axis=1) & ~np.isnan(real_xyz_mm).any(axis=1)
    
    if not np.any(valid_mask):
        print("Warning: No valid simulation results to compare. Returning large error.")
        return 1e9 # 返回一个很大的误差值

    filtered_sim_xyz = simulated_xyz_mm[valid_mask]
    filtered_real_xyz = real_xyz_mm[valid_mask]

    # 应用辨识出的z_offset到仿真Z值上
    # utils.py 中是 sim_z_mm_final = sim_z_mm_raw - Z_OFFSET_FROM_UTILS (e.g., 480)
    # 所以，如果我们辨识 Z_OFFSET_FROM_UTILS, 那么
    # error_z = (real_z - (sim_z_raw - Z_OFFSET_FROM_UTILS_PARAM))^2
    # 或者，real_z_shifted = real_z + Z_OFFSET_FROM_UTILS_PARAM
    # error_z = (real_z_shifted - sim_z_raw)^2
    # 我们选择第一种：在比较前调整仿真Z值
    # Sim_Z_corrected = Sim_Z_raw - identified_z_offset
    
    # 注意：utils.py中sim_z_mm的计算是 sim_z_m[2]*1000 - 480
    # 如果我们将辨识的 z_offset_mm 直接作为那个 "480"
    # 那么，sim_z_mm_final = sim_z_raw_m*1000 - params_dict['z_offset_mm']
    # simulated_xyz_mm 已经是 sim_z_raw_m*1000 了 (没有减去任何offset)
    
    errors_x = filtered_real_xyz[:, 0] - filtered_sim_xyz[:, 0]
    errors_y = filtered_real_xyz[:, 1] - filtered_sim_xyz[:, 1]
    errors_z = filtered_real_xyz[:, 2] - (filtered_sim_xyz[:, 2] - params_dict['z_offset_mm']) # 应用z_offset

    mse = np.mean(errors_x**2 + errors_y**2 + errors_z**2)
    
    # 打印当前参数和MSE，方便追踪
    param_str = ", ".join([f"{name}={val:.4f}" for name, val in params_dict.items()])
    print(f"Params: [{param_str}], MSE: {mse:.6f}")
    
    return mse

# --- 主优化流程 ---
if __name__ == "__main__":
    DATA_FILE_PATH = 'D:/data/load_data/random_data.xlsx' # 修改为你的路径
    SHEET_NAME = 'Sheet1'
    NUM_OPTIMIZATION_SAMPLES = 200  # 使用200条数据进行优化，可以根据速度调整

    # 1. cable_distance, 2. initial_length, 3. axial_strain_coefficient, 4. AXIAL_ACTION_SCALE, 5. z_offset_mm
    initial_params_guess = np.array([0.035, 0.12, -2.0, 0.8, 480.0])
    
    # 为参数设置合理的界限 (非常重要!)
    # (min_val, max_val) for each parameter
    bounds = [
        (0.035, 0.035),    # cable_distance (m)
        (0.12, 0.12),    # initial_length (m)
        (-20, 0.0),    # axial_strain_coefficient (k_strain)
        (0.5, 1.5),      # AXIAL_ACTION_SCALE
        (480, 480)   # z_offset_mm (mm)
    ]

    print("--- Starting Parameter Optimization ---")
    # Nelder-Mead 通常对噪声和不光滑的函数有较好鲁棒性，且不需要梯度
    # 但它可能收敛较慢，或陷入局部最优。可以尝试 'Powell' 或 'COBYLA'
    # 对于有界约束，'L-BFGS-B', 'TNC', 'SLSQP' 也可用，但它们期望函数较光滑
    # 'trust-constr' 也是一个不错的有界优化器

    # 在优化开始前初始化PyBullet
    init_pybullet_direct()

    try:
        result = minimize(
            cost_function,
            initial_params_guess,
            args=(DATA_FILE_PATH, SHEET_NAME, NUM_OPTIMIZATION_SAMPLES), # 传递给cost_function的额外参数
            method='Nelder-Mead', # 或者 'Powell', 'L-BFGS-B' (如果用L-BFGS-B, bounds参数格式不同)
            bounds=bounds, # Nelder-Mead 原生不支持bounds, 但scipy的实现会通过变量转换来处理
                           # 如果选择 SLSQP, TNC, L-BFGS-B, trust-constr, bounds可以直接使用
            options={'maxiter': 200, 'disp': True, 'adaptive': True} # adaptive for Nelder-Mead
        )

        identified_params_values = result.x
        final_mse = result.fun

        print("\n--- Optimization Finished ---")
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")
        print(f"Number of iterations: {result.nit}")
        print(f"Number of function evaluations: {result.nfev}")
        print(f"Final MSE: {final_mse:.6f}")
        print("Identified Parameters:")
        for name, val in zip(PARAM_NAMES, identified_params_values):
            print(f"  {name}: {val:.6f}")

    except Exception as e:
        print(f"An error occurred during optimization: {e}")
    finally:
        # 确保PyBullet在结束时关闭
        shutdown_pybullet()

    # 下一步：
    # 1. 使用找到的 identified_params_values 来更新你的 main_new.py 中的默认参数。
    # 2. 重新运行你的 LSTM_attention_model.py，它现在会使用基于优化后参数生成的仿真数据
    #    （你需要先用优化后的参数跑一遍main_new.py生成新的包含sim_X, sim_Y, sim_Z列的Excel文件）。
    # 3. 观察残差学习模型的性能是否有提升。