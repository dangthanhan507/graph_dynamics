import torch
from graph.dataset import TrackedDatasetBaseline
from graph.gnn import GNN_baseline
import argparse
import os
import open3d as o3d
import numpy as np 
from graph.visual_dataset import VisualizeEpisode

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_path', type=str, default='chkpt/official_model.pth')
    args = argparser.parse_args()
    
    state_history = 15
    dataset = TrackedDatasetBaseline('../dataset/rigid_d3_official_jayjun/',
                                     state_history=state_history,
                                     num_tracked_pts=100,
                                     topk_edge_threshold=10
    ) 
    visual_dataset = VisualizeEpisode('../dataset/rigid_d3_official_jayjun/episode_0')
    
    
    gnn = GNN_baseline(state_history_size=state_history,
                       num_vertices=101,
                       vertex_embedding_size=512,
                       edge_embedding_size=512,
                       num_features_vertex=512,
                       num_features_edge=512,
                       num_features_decode=512,
                       message_passing_steps=3).to('cuda')
    gnn.load_state_dict(torch.load(args.model_path, weights_only=True))
    gnn.eval()
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='visualizer')
    base_sphere_vertex = None
    pcl = None
    with torch.no_grad():
        for i in range(len(dataset)):
            data,label = dataset[i]
            for key in data:
                data[key] = data[key].to('cuda')
            out = gnn(**data)
            out = out.cpu().numpy()
            label = label.squeeze().cpu().numpy()
            
            pcd, ptsrgb = visual_dataset[i+state_history]
            
            workspace_bbox = np.array([[-0.3, 0.3],
                                       [0.3, 1.0],
                                       [-0.1, 1.0]])
            mask = (pcd[:,0] > workspace_bbox[0,0]) & (pcd[:,0] < workspace_bbox[0,1]) & \
                     (pcd[:,1] > workspace_bbox[1,0]) & (pcd[:,1] < workspace_bbox[1,1]) & \
                        (pcd[:,2] > workspace_bbox[2,0]) & (pcd[:,2] < workspace_bbox[2,1])
            pcd = pcd[mask, :]
            ptsrgb = ptsrgb[mask, :]
            
            # visualize spheres for out and label
            # do initial visualization
            if i == 0:
                origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
                vis.add_geometry(origin)
                out_spheres = []
                for i in range(out.shape[0]):
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005, resolution=10, create_uv_map=True)
                    sphere.compute_vertex_normals()
                    base_sphere_vertex = np.array(sphere.vertices)
                    sphere.vertices = o3d.utility.Vector3dVector(base_sphere_vertex)
                    out_spheres.append(sphere)
                    vis.add_geometry(sphere)
                # draw label spheres as yellow
                label_spheres = []
                for i in range(label.shape[0]):
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005, resolution=10, create_uv_map=True)
                    sphere.compute_vertex_normals()
                    sphere.vertices = o3d.utility.Vector3dVector(base_sphere_vertex)
                    sphere.paint_uniform_color([1, 1, 0])
                    label_spheres.append(sphere)
                    vis.add_geometry(sphere)
            if pcl is not None:
                # remove vis pcl and
                vis.remove_geometry(pcl)
            
            # only consider point clouds inside the workspace
            pcl = o3d.geometry.PointCloud()
            pcl.points = o3d.utility.Vector3dVector(pcd)
            pcl.colors = o3d.utility.Vector3dVector(ptsrgb/255)
            vis.add_geometry(pcl)
            
            for i in range(out.shape[0]):
                out_spheres[i].vertices = o3d.utility.Vector3dVector(base_sphere_vertex + out[i])
                label_spheres[i].vertices = o3d.utility.Vector3dVector(base_sphere_vertex + label[i])
                vis.update_geometry(out_spheres[i])
                vis.update_geometry(label_spheres[i])
                
            view_control = vis.get_view_control()
            # view_control.set_front([1.0, 1.0, 1.0])
            # view_control.set_lookat([0., 0.65, 0])
            # view_control.set_up([0, 0, 1])
            # view_control.set_zoom(0.400000000000008) 
            vis.poll_events()
            vis.update_renderer()
            vis.run()
        pass
    pass