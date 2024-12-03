import sys
sys.path.append('./')
import torch
from graph.dataset import TrackedDatasetBaseline
from torch.utils.data import DataLoader
from graph.gnn import GNN_baseline
def dataloader_wrapper(dataloader, name):
    cnt = 0
    while True:
        print(f'[{name}] epoch {cnt}')
        cnt += 1
        for data in dataloader:
            yield data

if __name__ == '__main__':
    batch_size = 10
    
    #load dataloader
    dataset = TrackedDatasetBaseline('../dataset/rigid_d3_official/', 
                                     state_history=15,
                                     num_tracked_pts=100,
                                     topk_edge_threshold=10
    )
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=8 # limit to number of CPU cores / supported threads
    )
    dataloader = dataloader_wrapper(dataloader, 'train')
    
    # load model
    gnn = GNN_baseline(state_history_size=15,
                       num_vertices=100 + 1,
                       vertex_embedding_size=512,
                       edge_embedding_size=512,
                       num_features_vertex=512,
                       num_features_edge=512,
                       num_features_decode=512,
                       message_passing_steps=3)
    
    data, label = next(dataloader)
    
    out = gnn(**data)
    print(out.shape)