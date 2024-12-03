import torch
from graph.dataset import TrackedDatasetBaseline
from graph.gnn import GNN_baseline
import argparse
import os
import open3d as o3d
import numpy as np 
from graph.visual_dataset import VisualizeEpisode
from graph.graph_gen import construct_edges_from_states

if __name__ == '__main__':
    visual_dataset = VisualizeEpisode('../dataset/rigid_d3_official_jayjun/episode_0')
    
    state_history = 15
    dataset = TrackedDatasetBaseline('../dataset/rigid_d3_official_jayjun/',
                                     state_history=state_history,
                                     num_tracked_pts=100,
                                     topk_edge_threshold=10
    ) 
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='visualizer')
    vis.get_render_option().line_width = 1
    base_sphere_vertex = None
    pcl = None
    line_set = None
    label_spheres = []
    start = 400
    for i in range(start, len(dataset)):
        data, label = dataset[i]
        label = label.squeeze().cpu().numpy()
        
        pcd, ptsrgb = visual_dataset[i+state_history]
        workspace_bbox = np.array([[-0.3, 0.3],
                                    [0.3, 1.0],
                                    [-0.1, 0.1]])
        mask = (pcd[:,0] > workspace_bbox[0,0]) & (pcd[:,0] < workspace_bbox[0,1]) & \
                    (pcd[:,1] > workspace_bbox[1,0]) & (pcd[:,1] < workspace_bbox[1,1]) & \
                    (pcd[:,2] > workspace_bbox[2,0]) & (pcd[:,2] < workspace_bbox[2,1])
        pcd = pcd[mask, :]
        ptsrgb = ptsrgb[mask, :]
        if i == start:
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            vis.add_geometry(origin)
            
            # draw label spheres as yellow
            for j in range(label.shape[0]):
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005, resolution=10, create_uv_map=True)
                sphere.compute_vertex_normals()
                base_sphere_vertex = np.array(sphere.vertices)
                sphere.vertices = o3d.utility.Vector3dVector(base_sphere_vertex)
                sphere.paint_uniform_color([1, 1, 0])
                label_spheres.append(sphere)
                vis.add_geometry(sphere)
                
            # add lines for graph
            line_set = o3d.geometry.LineSet()
            obj_mask = np.zeros((label.shape[0]), dtype=bool)
            tool_mask = np.zeros((label.shape[0]), dtype=bool)
            obj_mask[:label.shape[0]] = True
            label_tensor = torch.tensor(label)
            obj_mask = torch.tensor(obj_mask)
            tool_mask = torch.tensor(tool_mask)
            Rr, Rs = construct_edges_from_states(label_tensor, 0.05, obj_mask, tool_mask, topk=10)
            Rr = Rr.numpy()
            Rs = Rs.numpy()
            
            edges = np.zeros((Rr.shape[0], 2))
            for j in range(Rr.shape[0]):
                idx_Rr = np.where(Rr[j, :] == 1)[0][0]
                idx_Rs = np.where(Rs[j, :] == 1)[0][0]
                edges[j, 0] = idx_Rr
                edges[j, 1] = idx_Rs
            line_set.points = o3d.utility.Vector3dVector(label)
            line_set.lines = o3d.utility.Vector2iVector(edges)
            line_set.colors = o3d.utility.Vector3dVector(np.array([[1, 1, 0]]*edges.shape[0]))
            vis.add_geometry(line_set)
            
            
        if pcl is not None:
            # remove vis pcl and
            vis.remove_geometry(pcl)
        
        # only consider point clouds inside the workspace
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(pcd)
        pcl.colors = o3d.utility.Vector3dVector(ptsrgb/255)
        vis.add_geometry(pcl)
        
        line_set.points = o3d.utility.Vector3dVector(label)
        for i in range(label.shape[0]):
            label_spheres[i].vertices = o3d.utility.Vector3dVector(base_sphere_vertex + label[i])
            vis.update_geometry(label_spheres[i])
            
        view_control = vis.get_view_control()
        # view_control.set_front([1.0, 1.0, 1.0])
        # view_control.set_lookat([0., 0.65, 0])
        # view_control.set_up([0, 0, 1])
        # view_control.set_zoom(0.400000000000008) 
        vis.poll_events()
        vis.update_renderer()
        vis.run()