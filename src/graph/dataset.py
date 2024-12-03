import sys
sys.path.append('./')
import torch
from torch.utils.data.dataset import Dataset
import os
import pickle
from itertools import accumulate
import numpy as np
from graph.graph_gen import construct_edges_from_states
#TODO: while we are working with obj_kpts data, we need to add the scripts from D^3Fields that track keypoints and save to file

#ASSUMPTION: state has history of N for 15hz video (1 second of data)
#TODO: implement way to skip frames to be 5hz or 2hz
class TrackedDatasetBaseline(Dataset):
    def __init__(self, dataset_folder, num_tracked_pts = 100, state_history=15, adj_threshold = 0.1, topk_edge_threshold = 10):
        '''
            Dataset should consist of:
            ==========================
                -> Tracked 3D Points of Object
                -> Joint Data of Robot (for single-point tracking)
            
            dataset_name
                episode_i
                ├── obj_kypts
                    ├── 000000.pkl # 3D tracked points
                ├── end_effector_positions.npy
            
            NOTE: example of episode index (if state_history = 2)
            =====================================================
            episode_i:
            | frame 0 | frame 1 | frame 2 | frame 3 | frame 4 | frame 5 | frame 6 | ... | frame N - 2 | frame N - 1 | frame N |
            
            data_idx 0:  [frame 0, frame 1, frame 2]
            label_idx 0: [frame 3]
            
            data_idx N-2: [frame N-2, frame N-1, frame N]
            label_idx N-2: [frame N]
            
            
            NOTE: example of dataset index (if state_history = 2)
            =====================================================
            dataset:
            | episode_0 | episode_1 | episode_2 | episode_3 | ... | episode D | (D = 2)
            
            episode_0: 100 frames (98 datapoints)
            episode_1: 50  frames (48 datapoints)
            episode_2: 200 frames (198 datapoints)
            
            total_dataset: 350 frames (344 datapoints)
            
            
            KEY
            NOTE: in order to reconcile dataset indexing, this is our index scheme:
            ========================================================================
            scheme:
            | (frames in episode 0) - state_history | (frames in episode 1) - state_history | (frames in episode 2) - state_history | ... | (frames in episode D) - state_history |
        '''
        self.dataset_folder = dataset_folder
        self.state_history = state_history
        self.num_tracked_pts = num_tracked_pts
        self.adjacency_threshold = adj_threshold
        
        # if a node could only take on the top k edges
        # then max number of edges is topk_edge_threshold * num_vertices
        self.topk_edge_threshold = topk_edge_threshold
        self.max_edges = topk_edge_threshold * (num_tracked_pts + 1) # + 1 for end effector
        
        self.episode_lengths = [len(os.listdir(os.path.join(dataset_folder, episode_name,'camera_0/color'))) - state_history \
                                        for episode_name in os.listdir(dataset_folder) \
                                            if episode_name.startswith('episode')]
        self.episode_end_index = [i-1 for i in accumulate(self.episode_lengths)]
        
        # get length of dataset
        self.length_dataset = sum([len(os.listdir(os.path.join(dataset_folder, episode_name,'camera_0/color'))) - state_history \
                                        for episode_name in os.listdir(dataset_folder) \
                                            if episode_name.startswith('episode')])
    
    #NOTE: must overload these methods for Dataset
    def __getitem__(self, idx):
        
        # using episode_lengths, get which episode we are in for index
        # | epi 0: 0 -> N_1 | epi 1: N_1 + 1 -> N_1 + N_2 ...
        episode_idx = -1
        for i in range(len(self.episode_end_index)):
            if idx <= self.episode_end_index[i]:
                episode_idx = i
                break
        assert episode_idx != -1, f'Dataset Error: Index {idx} is out of bounds. Needs to satisfy 0 < x < {self.length_dataset}.'
        
        # get frame index in episode
        frame_idx = idx - (0 if episode_idx == 0 else self.episode_end_index[episode_idx-1] + 1)
        
        # get keypoints (frame_idx, frame_idx + 1, ... frame_idx + state_history) from files
        keypoints = torch.zeros(self.state_history + 1, self.num_tracked_pts, 3)
        for i in range(self.state_history + 1):
            filename = os.path.join(self.dataset_folder, f'episode_{episode_idx}', 'obj_kypts', f'{(frame_idx + i):06d}.pkl')
            pts = pickle.load(open(filename, 'rb'))[0]
            keypoints[i] = torch.tensor(pts)
        
        # get end effector positions from files
        ee_filename = os.path.join(self.dataset_folder, f'episode_{episode_idx}', 'end_effector_positions.npy')
        ee_positions = torch.tensor(np.load(ee_filename)[frame_idx:frame_idx + self.state_history + 1, :])
        '''
            NOTE: structure of keypoints array
            ===================================
            | kypts 0 | ... | kypts state_history - 1 | kypts state_history |
            
            kypts[0] -> kypts[state_history - 1] is input data
            label is kypts[state_history]
        '''
        vertices = torch.concatenate([keypoints, ee_positions[:,None,:]], dim=1) # (state_history + 1, num_vertices, 3)
        
        object_mask = torch.zeros((vertices.shape[1]), dtype=bool)
        object_mask[:-1] = True
        
        tool_mask = torch.zeros((vertices.shape[1]), dtype=bool)
        tool_mask[-1] = True
        
        vertices_history = vertices[:-1, :, :]
        future_vertices   = vertices[-1,  :, :][None, :, :]
        
        
        # construct graph from last particle in horizon
        Rr, Rs = construct_edges_from_states(vertices_history[-1], self.adjacency_threshold, object_mask, tool_mask, topk=self.topk_edge_threshold)
        
        # edge_ij = (node_sender, node_receiver) = (node_i, node_j)
        # points from node_i to node_j
        # Rr (n_edge, num_vertices) one-hot for which vertices is receiver node for each edge
        # Rs (n_edge, num_vertices) one-hot for which vertices is sender node for each edge
        
        # attrs or category vector between tool and object
        onehot_vertices = torch.zeros((vertices.shape[1], 2))
        onehot_vertices[-1, 1]  = 1 # label last vertices as tool
        onehot_vertices[:-1, 0] = 1 # label rest of vertices as object
        
        # category vector on whether edge is object-object or object-tool relation
        # ASSUMPTION: there are no tool-tool relations, anything related to tool is object-tool relation
        onehot_edges     = torch.zeros((Rr.shape[0], 2)) # (n_edge, 2)
        
        # mask for which receiver and sender one-hot contains a tool
        Rr_mask = (Rr[:, -1] == 1)
        Rs_mask = (Rs[:, -1] == 1)
        edge_mask = (Rr_mask | Rs_mask)
        
        # column 0 is object-object, column 1 is object-tool
        onehot_edges[edge_mask, 1] = 1
        
        # edge encoder takes in a fixed dimension, the # of edges vary wildly, so pad the onehot_edges
        if onehot_edges.shape[0] < self.max_edges:
            onehot_edges = torch.cat([onehot_edges, torch.zeros((self.max_edges - onehot_edges.shape[0], 2))], dim=0)
        if Rr.shape[0] < self.max_edges:
            Rr = torch.cat([Rr, torch.zeros((self.max_edges - Rr.shape[0], Rr.shape[1]))], dim=0)
            Rs = torch.cat([Rs, torch.zeros((self.max_edges - Rs.shape[0], Rs.shape[1]))], dim=0)
        
        graph_data = {
            'vertices_history': vertices_history, # (state_history, num_vertices, 3)
            'onehot_vertices': onehot_vertices, # (num_vertices, 2)
            'onehot_edges': onehot_edges, # (n_edge, 2)
            'Rr': Rr, # (n_edge, num_vertices)
            'Rs': Rs, # (n_edge, num_vertices)
        }
        return graph_data, future_vertices
        
    def __len__(self):
        return self.length_dataset
    
if __name__ == '__main__':
    dataset = TrackedDatasetBaseline('../dataset/rigid_d3_official/', state_history=15)
    
    print(dataset.episode_end_index)
    print(len(dataset))
    print(dataset[0])
    print(dataset[0]['onehot_edges'].shape)