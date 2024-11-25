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
from hardware.cameras import Cameras, depth2pcd
#NOTE: test getting point clouds from realsense camera

class CamerasToPointCloud(LeafSystem):
    def __init__(self, cameras: Cameras):
        LeafSystem.__init__(self)
        self.cameras = cameras
        self.Ks = cameras.get_intrinsics()
        self.n_cam = self.cameras.n_fixed_cameras
        
        self.DeclareAbstractOutputPort("point_cloud", lambda: Value(PointCloud()), self.pclOut)
    def pclOut(self, context, output):
        obs = self.cameras.get_obs(get_color=True, get_depth=True)
        color0 = obs['color_0'][-1]
        depth0 = obs['depth_0'][-1]
        
        
        # start = time.time()
        pts3d, rgb = depth2pcd(depth0, self.Ks[0], color0[:,:,::-1])
        pcl = PointCloud(new_size = pts3d.shape[0], fields= Fields(BaseField.kXYZs | BaseField.kRGBs))
        mutable_rgb = pcl.mutable_rgbs().T
        mutable_xyz = pcl.mutable_xyzs().T
        mutable_rgb[:] = rgb
        mutable_xyz[:] = pts3d
        pcl.VoxelizedDownSample(voxel_size=0.03, parallelize=True)
        # delta = time.time() - start
        # print(delta)
        
        
        output.set_value(pcl)
        pass
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
        process_depth=True
    )
    cameras.start(exposure_time=10)
    
    builder = DiagramBuilder()
    
    
    cam2pcl = builder.AddSystem(CamerasToPointCloud(cameras))
    meshcat_pcl_vis = builder.AddSystem(MeshcatPointCloudVisualizer(meshcat, path="/drake", publish_period=0.01))
    
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
    
    