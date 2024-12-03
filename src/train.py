import torch
from graph.dataset import TrackedDatasetBaseline
from torch.utils.data import DataLoader, random_split
from graph.gnn import GNN_baseline
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random
import argparse
import os

def dataloader_wrapper(dataloader, name):
    cnt = 0
    while True:
        print(f'[{name}] epoch {cnt}')
        cnt += 1
        for data in dataloader:
            yield data

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--chkpt_path', type=str, default='chkpt')
    argparser.add_argument('--model_file', type=str, default='model.pth')
    args = argparser.parse_args()
    set_seed(42)
    batch_size = 256
    total_epochs = 100
    state_history = 15
    ckpt_every_k_epochs = 10
    #load dataloader
    dataset = TrackedDatasetBaseline('../dataset/rigid_d3_official/', 
                                     state_history=state_history,
                                     num_tracked_pts=100,
                                     topk_edge_threshold=10
    )
    
    ratio_train = 0.8
    train_dataset, val_dataset = random_split(dataset, [int(len(dataset)*ratio_train), len(dataset) - int(len(dataset)*ratio_train)])
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=8
    )
    val_dataloader   = DataLoader(val_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=8
    )
    
    # load model
    gnn = GNN_baseline(state_history_size=state_history,
                       num_vertices=100 + 1,
                       vertex_embedding_size=512,
                       edge_embedding_size=512,
                       num_features_vertex=512,
                       num_features_edge=512,
                       num_features_decode=512,
                       message_passing_steps=3).to('cuda')
    gnn.train()
    optimizer = torch.optim.Adam(gnn.parameters(), lr=1e-5)
    
    train_losses = []
    validation_losses = []
    for epoch in range(total_epochs):
        print("Epoch:", epoch)
        for data,label in tqdm(train_dataloader):
            optimizer.zero_grad()
            for key in data:
                data[key] = data[key].to('cuda')
            label = label.to('cuda')
            out = gnn(**data)
            loss = F.mse_loss(out * 1000.0, label.squeeze() * 1000.0)
            loss.backward()
            optimizer.step()
            
            del data, label, loss
        
        train_loss = 0
        validation_loss = 0
        num_loops_train = 0
        num_loop_val = 0
        with torch.no_grad():
            for data,label in train_dataloader:
                num_loops_train += 1
                for key in data:
                    data[key] = data[key].to('cuda')
                label = label.to('cuda')
                out = gnn(**data)
                train_loss += F.mse_loss(out * 1000.0, label.squeeze() * 1000.0)
                
                del data, label
            
            for data,label in val_dataloader:
                num_loop_val += 1
                for key in data:
                    data[key] = data[key].to('cuda')
                label = label.to('cuda')
                out = gnn(**data)
                validation_loss += F.mse_loss(out * 1000.0, label.squeeze() * 1000.0)
                
                del data, label
        print("Train Loss:", train_loss / num_loops_train)
        print("Validation Loss:", validation_loss / num_loop_val)
        train_loss_avg = train_loss / num_loops_train
        validation_loss_avg = validation_loss / num_loop_val
        train_losses.append(train_loss_avg.cpu().numpy())
        validation_losses.append(validation_loss_avg.cpu().numpy())
    
    np.save(os.path.join(args.chkpt_path,'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(args.chkpt_path,'validation_losses.npy'), np.array(validation_losses))
    
    # save to pth
    os.makedirs(args.chkpt_path, exist_ok=True)
    torch.save(gnn.state_dict(), os.path.join(args.chkpt_path, args.model_file))