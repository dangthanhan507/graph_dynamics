import numpy as np
from pydrake.all import (
    RigidTransform,
    DiagramBuilder,
    Simulator,
    MeshcatPointCloudVisualizer,
    StartMeshcat,
    CameraInfo,
    LeafSystem,
    PointCloud,
    BaseField,
    Fields,
    Value
)
import time
from hardware.cameras import Cameras, depth2pcd, load_extrinsics
#NOTE: test getting point clouds from realsense camera

class CamerasToPointCloud(LeafSystem):
    def __init__(self, cameras: Cameras):
        LeafSystem.__init__(self)
        self.cameras = cameras
        self.Ks = cameras.get_intrinsics()
        self.n_cam = self.cameras.n_fixed_cameras
        self.extrinsics = cameras.extrinsics
        self.cam2worlds = np.array([np.linalg.inv(self.extrinsics[i]) for i in range(self.n_cam)])
        
        self.DeclareAbstractOutputPort("point_cloud", lambda: Value(PointCloud()), self.pclOut)
    def pclOut(self, context, output):
        obs = self.cameras.get_obs(get_color=True, get_depth=True)
        color0 = obs['color_0'][-1]
        depth0 = obs['depth_0'][-1]
        
        color1 = obs['color_1'][-1]
        depth1 = obs['depth_1'][-1]
        
        color2 = obs['color_2'][-1]
        depth2 = obs['depth_2'][-1]
        
        color3 = obs['color_3'][-1]
        depth3 = obs['depth_3'][-1]
        
        # start = time.time()
        pts3d_0, rgb_0 = depth2pcd(depth0, self.Ks[0], color0[:,:,::-1])
        pts3d_1, rgb_1 = depth2pcd(depth1, self.Ks[1], color1[:,:,::-1])
        pts3d_2, rgb_2 = depth2pcd(depth2, self.Ks[2], color2[:,:,::-1])
        pts3d_3, rgb_3 = depth2pcd(depth3, self.Ks[3], color3[:,:,::-1])
        
        Rot0 = self.cam2worlds[0][:3,:3]
        t0 = self.cam2worlds[0][:3,3]
        pts3d_0 = (Rot0 @ pts3d_0.T).T + t0
        
        
        Rot1 = self.cam2worlds[1][:3,:3]
        t1 = self.cam2worlds[1][:3,3]
        pts3d_1 = (Rot1 @ pts3d_1.T).T + t1
        
        Rot2 = self.cam2worlds[2][:3,:3]
        t2 = self.cam2worlds[2][:3,3]
        pts3d_2 = (Rot2 @ pts3d_2.T).T + t2
        
        Rot3 = self.cam2worlds[3][:3,:3]
        t3 = self.cam2worlds[3][:3,3]
        pts3d_3 = (Rot3 @ pts3d_3.T).T + t3
        
        pts3d = np.concatenate([pts3d_0, pts3d_1, pts3d_2, pts3d_3], axis=0)
        rgb = np.concatenate([rgb_0, rgb_1, rgb_2, rgb_3], axis=0)
        
        pcl = PointCloud(new_size = pts3d.shape[0], fields= Fields(BaseField.kXYZs | BaseField.kRGBs))
        pcl.mutable_rgbs()[:] = rgb.T
        pcl.mutable_xyzs()[:] = pts3d.T
        
        output.set_value(pcl.VoxelizedDownSample(voxel_size=1e-2, parallelize=False))
        # delta = time.time() - start
        # print(delta)
        
if __name__ == '__main__':
    print("For Jayjun-san")
    
    meshcat = StartMeshcat()
    cameras = Cameras(
        WH=[640, 480],
        capture_fps=15,
        obs_fps=30,
        n_obs_steps=2,
        enable_color=True,
        enable_depth=True,
        process_depth=True,
        extrinsic_path='../config/camera_extrinsics.json'
    )
    cameras.start(exposure_time=10)
    
    builder = DiagramBuilder()
    
    
    cam2pcl = builder.AddSystem(CamerasToPointCloud(cameras))
    meshcat_pcl_vis = builder.AddSystem(MeshcatPointCloudVisualizer(meshcat, path="/drake", publish_period=1e-4))
    
    builder.Connect(
        cam2pcl.get_output_port(0),
        meshcat_pcl_vis.get_input_port(0)
    )
    
    diagram = builder.Build()
    
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    
    simulator.set_target_realtime_rate(1.0)
    
    # test
    meshcat.AddButton("Stop Simulation", "Escape")
    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        simulator.AdvanceTo(simulator.get_context().get_time() + 2.0)
    meshcat.DeleteButton("Stop Simulation")
    
    