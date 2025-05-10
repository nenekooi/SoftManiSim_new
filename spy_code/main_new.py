import sys
import os
import numpy as np
import pybullet as p
import pybullet_data
import time
import math
from pprint import pprint
import pandas as pd
import utils

softmanisim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(softmanisim_path)
from visualizer.spy_visualizer import ODE


if __name__ == "__main__":

    print("--- 设置参数 ---")
    DATA_FILE_PATH = 'D:/data/load_data/random_data.xlsx'
    # DATA_FILE_PATH = 'D:/data/load_data/circle.xlsx'
    SHEET_NAME = 'Sheet1'
    # OUTPUT_RESULTS_PATH = 'D:/data/save_data/5(u_new_4,cab=0.035,k=-20,a=1).xlsx'
    OUTPUT_RESULTS_PATH = 'D:/data/save_data/aaa2(u_new_5,cab=0.035,k=-2,a=0.8).xlsx'
    
    # ---机器人参数---
    num_cables = 3 
    cable_distance = 0.035
    initial_length = 0.12
    number_of_segment = 1
    L0_seg = initial_length / number_of_segment
    print(f"机器人参数: L0={initial_length:.4f}m, d={cable_distance:.4f}m")
    axial_strain_coefficient = -2
    AXIAL_ACTION_SCALE = 0.8

    #---机器人可视化参数---
    body_color = [1, 0.0, 0.0, 1]
    head_color = [0.0, 0.0, 0.75, 1]
    body_sphere_radius = 0.02
    number_of_sphere = 30
    my_sphere_radius = body_sphere_radius
    my_number_of_sphere = number_of_sphere
    my_head_color = head_color

    # --- 加载数据 ---
    absolute_lengths_mm, real_xyz_mm, dl_sequence_m, L0_cables_mm = utils.load_and_preprocess_data(DATA_FILE_PATH, SHEET_NAME)

    # --- 初始化结果存储 ---
    results_data = utils.initialize_results_storage()

    # --- PyBullet 初始化 ---
    print("--- 初始化 PyBullet ---")
    simulationStepTime = 0.0001
    physicsClientId = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(simulationStepTime)
    planeId = p.loadURDF("plane.urdf")
    print(f"加载 plane.urdf, ID: {planeId}")
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=45, cameraPitch=-20, cameraTargetPosition=[0, 0, 0.3])
    print("[信息] 已设置相机视角。")

    # --- 初始化 ODE 对象 ---
    print("--- 初始化 ODE 对象 ---")
    my_ode = ODE()
    my_ode.l0 = initial_length
    my_ode.d = cable_distance
    print(f"ODE 已初始化 L0={my_ode.l0:.4f}m, d={my_ode.d:.4f}m")

    # --- 计算初始形态 ---
    print("--- 计算初始形状 (dl=0) ---")
    act0_segment = np.zeros(3)
    my_ode._reset_y0()
    my_ode.updateAction(act0_segment)
    sol0 = my_ode.odeStepFull()
    print(f"初始形状计算完成。") 

    # --- 设置基座 ---
    base_pos = np.array([0, 0, 0.6])
    base_ori_euler = np.array([-math.pi / 2.0, 0, math.pi / 4.7])
    base_ori = p.getQuaternionFromEuler(base_ori_euler)
    print(f"[设置] 基座世界坐标: {base_pos}")
    print(f"[设置] 基座世界姿态 (Euler): {base_ori_euler}")
    radius = my_sphere_radius

    # --- 创建 PyBullet 形状 ---
    print("--- 创建 PyBullet 形状 ---")
    shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    visualShapeId = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=body_color)
    visualShapeId_tip_body = p.createVisualShape(p.GEOM_SPHERE, radius=radius+0.0025, rgbaColor=my_head_color)
    visualShapeId_tip_gripper = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.002, 0.01, 0.002], rgbaColor=head_color)

    # --- 创建 PyBullet 物体 ---
    print("--- 创建 PyBullet 物体 ---")
    idx0 = np.linspace(0, sol0.shape[1] - 1, my_number_of_sphere, dtype=int)
    positions0_local = [(sol0[0, i], sol0[2, i], sol0[1, i]) for i in idx0]
    my_robot_bodies = []
    for i, pos_local in enumerate(positions0_local):
        pos_world, ori_world = p.multiplyTransforms(base_pos, base_ori, pos_local, [0,0,0,1])
        my_robot_bodies.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visualShapeId, basePosition=pos_world, baseOrientation=ori_world))

    ori_tip_local, _ = utils.calculate_orientation(positions0_local[-3], positions0_local[-1]) 
    pos_tip_world, ori_tip_world = p.multiplyTransforms(base_pos, base_ori, positions0_local[-1], ori_tip_local)
    my_robot_bodies.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visualShapeId_tip_body, basePosition=pos_tip_world, baseOrientation=ori_tip_world))
    gripper_offset1 = [0, 0.01, 0]
    pos1, _ = p.multiplyTransforms(pos_tip_world, ori_tip_world, gripper_offset1, [0,0,0,1])
    my_robot_bodies.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visualShapeId_tip_gripper, basePosition=pos1, baseOrientation=ori_tip_world))
    gripper_offset2 = [0,-0.01, 0]
    pos2, _ = p.multiplyTransforms(pos_tip_world, ori_tip_world, gripper_offset2, [0,0,0,1])
    my_robot_bodies.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visualShapeId_tip_gripper, basePosition=pos2, baseOrientation=ori_tip_world))
    print(f"总共创建了 {len(my_robot_bodies)} 个物体。")

    print("--- 初始化完成, 开始主仿真循环 ---")

    # --- 主循环 (改为 for 循环) ---
    num_data_rows = len(dl_sequence_m)
    print(f"将按顺序应用 {num_data_rows} 组绳长变化量数据。")
    for current_row_index in range(num_data_rows):
        dl_segment = dl_sequence_m[current_row_index]
        current_cblen_mm = absolute_lengths_mm[current_row_index]
        current_real_xyz_mm = real_xyz_mm[current_row_index]

        # --- 计算新形态 ---
        my_ode._reset_y0()
        ux, uy = utils.calculate_curvatures_from_dl_v4(dl_segment, cable_distance, L0_seg, AXIAL_ACTION_SCALE)
        avg_dl = np.mean(dl_segment)
        commanded_length_change = avg_dl * AXIAL_ACTION_SCALE
        
        my_ode.set_kinematic_state_spy(commanded_length_change, ux, uy)
        sol = my_ode.odeStepFull()

        # --- 更新可视化并获取仿真末端位置 ---
        pos_tip_world_m = np.array([np.nan, np.nan, np.nan])
        if sol is not None and sol.shape[1] >= 3:
            idx = np.linspace(0, sol.shape[1] -1, my_number_of_sphere, dtype=int)
            positions_local = [(sol[0, i], sol[2, i], sol[1, i]) for i in idx]
            num_bodies_total = len(my_robot_bodies)
            num_tip_bodies = 3
            num_body_spheres = num_bodies_total - num_tip_bodies
            num_points_available = len(positions_local)
            num_spheres_to_update = min(num_body_spheres, num_points_available)

            for i in range(num_spheres_to_update):
                pos_world, ori_world = p.multiplyTransforms(base_pos, base_ori, positions_local[i], [0,0,0,1])
                p.resetBasePositionAndOrientation(my_robot_bodies[i], pos_world, ori_world)

            ori_tip_local, _ = utils.calculate_orientation(positions_local[-3], positions_local[-1])
            pos_tip_world_tuple, ori_tip_world = p.multiplyTransforms(base_pos, base_ori, positions_local[-1], ori_tip_local)
            pos_tip_world_m = np.array(pos_tip_world_tuple)

            p.resetBasePositionAndOrientation(my_robot_bodies[-3], pos_tip_world_m, ori_tip_world)
            gripper_offset1 = [0, 0.01, 0]
            pos1, _ = p.multiplyTransforms(pos_tip_world_m, ori_tip_world, gripper_offset1, [0,0,0,1])
            p.resetBasePositionAndOrientation(my_robot_bodies[-2], pos1, ori_tip_world)
            gripper_offset2 = [0,-0.01, 0]
            pos2, _ = p.multiplyTransforms(pos_tip_world_m, ori_tip_world, gripper_offset2, [0,0,0,1])
            p.resetBasePositionAndOrientation(my_robot_bodies[-1], pos2, ori_tip_world)



        utils.append_result(results_data, current_cblen_mm, current_real_xyz_mm, pos_tip_world_m)


        p.stepSimulation()
        time.sleep(simulationStepTime)


    print("[信息] 所有数据已播放完毕。") 

    # --- 仿真结束，清理 ---
    print("[信息] 断开 PyBullet 连接。")
    p.disconnect(physicsClientId)

    # --- 保存结果 ---
    # utils.save_results_to_excel(results_data, OUTPUT_RESULTS_PATH)

    print("--- 仿真结束 ---")