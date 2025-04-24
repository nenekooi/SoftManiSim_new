
import pybullet as p
import os
from datetime import datetime
import numpy as np



class Camera:
    def __init__(self, cam_pos, cam_target, near, far, size, fov):
        self.x, self.y, self.z = cam_pos
        self.x_t, self.y_t, self.z_t = cam_target
        self.width, self.height = size
        self.near, self.far = near, far
        self.fov = fov
        
        aspect = self.width / self.height
        fov_rad = self.fov * (np.pi / 180.0)

        self.fy = (self.height / 2.0) / np.tan(fov_rad / 2.0)
        self.fx = self.fy * aspect
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        
        self.translation = cam_pos
        self.rotation = p.getQuaternionFromEuler([np.pi, 0, 0])


        aspect = self.width / self.height
        self.projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
        self.view_matrix = p.computeViewMatrix(cam_pos, cam_target, [0, 1, 0])

        self.rec_id = None

    def get_cam_img(self):
        """
        Method to get images from camera
        return:
        rgb
        depth
        segmentation mask
        """
        # Get depth values using the OpenGL renderer
        _w, _h, rgb, depth, seg = p.getCameraImage(self.width, self.height,
                                                   self.view_matrix, self.projection_matrix,
                                                   )
        return rgb[:, :, 0:3], depth, seg
    
    
    def pixel_to_world(self, u, v, Z):
        """
        Convert a pixel location (u, v) with a known depth Z in the camera's view
        to world coordinates using PyBullet transforms.

        Parameters:
            u (float): Pixel x-coordinate
            v (float): Pixel y-coordinate
            Z (float): Depth value associated with pixel (u,v) from the camera

        Returns:
            np.array: A (3,) numpy array representing the 3D position in world coordinates.
        """

        # Extract camera intrinsics
        fx = self.fx
        fy = self.fy
        cx = self.cx
        cy = self.cy

        # Back-project from pixel to camera coordinates
        X_cam = (u - cx) * Z / fx
        Y_cam = (v - cy) * Z / fy
        Z_cam = Z
        cam_world_pos, cam_world_orn = self.translation, self.rotation
        # self.extrinsic presumably is from world to camera if it's used like a view matrix.
        # If so, invert it:
        cam_to_world_pos, cam_to_world_orn = p.invertTransform(cam_world_pos, cam_world_orn)

        world_pos, world_ori = p.multiplyTransforms(cam_to_world_pos, cam_to_world_orn, [X_cam, Y_cam, Z_cam], [0, 0, 0, 1])
        return world_pos, world_ori



    def start_recording(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file = f'{save_dir}/{now}.mp4'

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.rec_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, file)

    def stop_recording(self):
        p.stopStateLogging(self.rec_id)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

