import numpy as np
from graph.graph_gen import fps, pad, construct_edges_from_states
import open3d as o3d
import torch

#NOTE: we will get 3d points from the object, then get 3d points from forward kinematics of robot as tool points
if __name__ == '__main__':
    # load rgb
    # load pts3d
    with open('rgb.npy', 'rb') as f:
        rgb = np.load(f)
    with open('pcd.npy', 'rb') as f:
        pts3d = np.load(f)
        
    max_nobj = 100
    fps_idx_list = fps(pts3d, max_nobj=max_nobj, fps_radius_range=0.01, verbose=False)
    
    obj_kp = pts3d[fps_idx_list, :]
    rgb_kp = rgb[fps_idx_list, :]
    
    obj_kp_fps = pad(obj_kp, max_nobj, dim=0)
    rgb_kp_fps = pad(rgb_kp, max_nobj, dim=0)
    
    
    # visualize pts3d first
    
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pts3d)
    # pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
    # # o3d.visualization.draw_geometries([pcd])
    
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(obj_kp)
    # pcd.colors = o3d.utility.Vector3dVector(rgb_kp / 255.0)
    # # o3d.visualization.draw_geometries([pcd])
    
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(obj_kp_fps)
    # pcd.colors = o3d.utility.Vector3dVector(rgb_kp_fps / 255.0)
    # o3d.visualization.draw_geometries([pcd])
    
    obj_mask = np.zeros((obj_kp_fps.shape[0]), dtype=bool)
    obj_mask[:obj_kp.shape[0]] = True # everything but padding is object
    tool_mask = np.zeros((obj_kp_fps.shape[0]), dtype=bool) # no tools in this case
    adj_thresh = 0.1
    
    obj_mask = torch.tensor(obj_mask)
    tool_mask = torch.tensor(tool_mask)
    obj_kp_fps = torch.tensor(obj_kp_fps)
    
    # construct edges is torch function
    Rr, Rs = construct_edges_from_states(obj_kp_fps, adj_thresh, obj_mask, tool_mask, topk=10)
    obj_kp_fps_npy = obj_kp_fps.numpy()
    
    # visualize edges on o3d lineset
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obj_kp_fps_npy)
    # pcd.colors = o3d.utility.Vector3dVector(rgb_kp_fps / 255.0)
    line_set = o3d.geometry.LineSet()
    edges = np.zeros((Rr.shape[0], 2))
    for i in range(Rr.shape[0]):
        idx_Rr = np.where(Rr[i, :] == 1)[0][0]
        idx_Rs = np.where(Rs[i, :] == 1)[0][0]
        edges[i, 0] = idx_Rr
        edges[i, 1] = idx_Rs
    line_set.points = o3d.utility.Vector3dVector(obj_kp_fps_npy)
    line_set.lines = o3d.utility.Vector2iVector(edges)
    o3d.visualization.draw_geometries([pcd, line_set])