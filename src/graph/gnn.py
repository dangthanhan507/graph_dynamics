import torch
import torch.nn as nn

# NOTE: YOINKED from gs-dynamics
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )
        self.output_size = output_size

    def forward(self, x):
        s = x.size()
        x = self.model(x.view(-1, s[-1]))
        return x.view(list(s[:-1]) + [self.output_size])
    
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Decoder, self).__init__()

        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.output_size = output_size

    def forward(self, x):
        s_x = x.size()

        x = x.view(-1, s_x[-1])
        x = self.relu(self.linear_0(x))

        x = self.relu(self.linear_1(x))
        return self.linear_2(x).view(list(s_x[:-1]) + [self.output_size])

class Propagator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Propagator, self).__init__()

        self.linear = nn.Linear(input_size, output_size)
        # self.linear_2 = nn.Linear(output_size, output_size)
        self.relu = nn.ReLU()
        self.output_size = output_size

    def forward(self, x, res=None):
        s_x = x.size()

        x = self.linear(x.view(-1, s_x[-1]))
        # x = self.relu(x)
        # x = self.linear_2(x)

        if res is not None:
            s_res = res.size()
            x += res.view(-1, s_res[-1])

        x = self.relu(x).view(list(s_x[:-1]) + [self.output_size])
        return x

class GNN_baseline(nn.Module):
    def __init__(self,  
                 state_history_size    = 15,
                 num_vertices          = 100,
                 max_edges             = 1010,
                 vertex_embedding_size = 128,
                 edge_embedding_size   = 128,
                 num_features_vertex   = 512,
                 num_features_edge     = 512,
                 num_features_decode   = 512,
                 message_passing_steps = 5):
        super(GNN_baseline, self).__init__()
        self.state_history_size = state_history_size
        
        self.vertex_embedding_size = vertex_embedding_size
        self.edge_embedding_size   = edge_embedding_size
        
        self.num_features_vertex   = num_features_vertex
        self.num_features_edge     = num_features_edge
        self.num_features_decode   = num_features_decode
        self.message_passing_steps = message_passing_steps
        
        # clamp delta motion
        self.clamp_velocity = 100.0
        
        # vertex encoder takes in vertices_history + onehot_vertices + velocity of vertices
        vertex_encoder_input_dim   = 3 * (state_history_size * num_vertices) + \
                                     3 * (state_history_size - 1) * num_vertices + \
                                     2 * num_vertices
        self.vertex_encoder        = Encoder(vertex_encoder_input_dim, self.num_features_vertex, self.vertex_embedding_size)
        
        # edge encoder takes in onehot_edges + padding + position difference of nodes of edge
        edge_encoder_input_dim     = 2 * max_edges + 3 * max_edges
        self.edge_encoder          = Encoder(edge_encoder_input_dim, self.num_features_edge, self.edge_embedding_size)
        
        # edge propagation
        edge_propagator_dim        = max_edges * edge_embedding_size + max_edges * vertex_embedding_size * 2
        self.edge_propagator       = Propagator(edge_propagator_dim, self.edge_embedding_size)
        
        # vertex propagation
        vertex_propagator_dim      = num_vertices * vertex_embedding_size + num_vertices * edge_embedding_size
        self.vertex_propagator     = Propagator(vertex_propagator_dim, self.vertex_embedding_size)
        
        # decoder go from vertex_embedding_size to 3
        self.decoder               = Decoder(self.vertex_embedding_size, self.num_features_decode, 3)
    def forward(self, vertices_history, onehot_vertices, onehot_edges, Rr, Rs):
        '''
        vertices_history: (state_history, num_vertices, 3)
        onehot_vertices: (num_vertices, 2)
        onehot_edges: (n_edge + pad, 2)
        Rr: (n_edge, num_vertices)
        Rs: (n_edge, num_vertices)
        
        NOTE: also these can have batch dimension
        '''
        is_batch = (len(vertices_history.shape) == 4)
        
        dim_concat_vertex = 2 if is_batch else 1 # if batch dimension is present -> 2
        velocity = (vertices_history[1:] - vertices_history[:-1]) # (state_history - 1, num_vertices, 3)
        
        # vertex_encoder_input dim will be (num_vertices, -1)
        vertices_history_flat = vertices_history.permute(0, 2, 1, 3) if is_batch else vertices_history.permute(1, 0, 2)
        vertices_history_flat = vertices_history_flat.view(vertices_history_flat.shape[0], vertices_history_flat.shape[1], -1) \
                                    if is_batch else vertices_history_flat.view(vertices_history_flat.shape[0], -1)
        
        velocity_flat         = velocity.permute(0, 2, 1, 3) if is_batch else velocity.permute(1, 0, 2)
        velocity_flat         = velocity_flat.view(velocity_flat.shape[0], velocity_flat.shape[1], -1) \
                                    if is_batch else velocity_flat.view(velocity_flat.shape[0], -1)
        
        dim_concat_vertex     = 2 if is_batch else 1
        vertex_encoder_input  = torch.cat([vertices_history_flat, velocity_flat, onehot_vertices], dim=dim_concat_vertex)
        
        '''
            Rr @ latest_state is weird, so this is an example of what's happening
            
            Let n1, n2, n3 denote 3 nodes
            Say edge set = {n1 -> n2, 
                            n1 -> n3,
                            n2 -> n3, 
                            n3 -> n1}
            Rs = [[1 0 0],
                  [1 0 0],
                  [0 1 0],
                  [0 0 1]]
                  
            Rr = [[0 1 0],
                  [0 0 1],
                  [1 0 0],
                  [1 0 0]]
            
            let position of n1 be (x1, y1, z1),
                position of n2 be (x2, y2, z2),
                position of n3 be (x3, y3, z3)
            
            latest_state = [[x1 y1 z1],
                            [x2 y2 z2],
                            [x3 y3 z3]]
            
            Rs @ latest_state = [[x1 y1 z1],
                                 [x1 y1 z1],
                                 [x2 y2 z2],
                                 [x3 y3 z3]]
            So Rs @ latest_state gives the positions of the sender nodes for each edge
        '''
        latest_state = vertices_history[:, -1, :, :] if is_batch else vertices_history[-1, :, :]
        state_receivers = Rr.bmm(latest_state) if is_batch else Rr @ latest_state # ((B,), n_edge + pad, 3)
        state_senders   = Rs.bmm(latest_state) if is_batch else Rs @ latest_state 
        
        edge_position_diff = state_receivers - state_senders # ((B,), n_edge + pad, 3)
        dim_concat_edge = 2 if is_batch else 1
        edge_encoder_input = torch.cat([onehot_edges, edge_position_diff], dim=dim_concat_edge)
        
        # get embeddings
        vertex_embedding = self.vertex_encoder(vertex_encoder_input) # ((B,), num_vertices, vertex_embedding_size)
        edge_embedding = self.edge_encoder(edge_encoder_input)       # ((B,), n_edge + pad, edge_embedding_size)
        
        Rr_agg = Rr.permute(0, 2, 1) if is_batch else Rr.permute(1, 0) # ((B,), num_vertices, n_edge + pad)
        
        #propagation steps
        step0_vertex_embedding = vertex_embedding
        for _ in range(self.message_passing_steps):
        
            # edge propagation,
            receiver_vertex_embedding = Rr.bmm(vertex_embedding) if is_batch else Rr @ vertex_embedding # ((B,), n_edge + pad, vertex_embedding_size)
            sender_vertex_embedding   = Rs.bmm(vertex_embedding) if is_batch else Rs @ vertex_embedding # ((B,), n_edge + pad, vertex_embedding_size)
            edge_prop_input           = torch.cat([edge_embedding, receiver_vertex_embedding, sender_vertex_embedding], dim=dim_concat_edge)
            edge_embedding            = self.edge_propagator(edge_prop_input) # ((B,), n_edge + pad, edge_embedding_size)
            
            # vertex propagation
            receiver_edge_embeddings  = Rr_agg.bmm(edge_embedding) if is_batch else Rr_agg @ edge_embedding # ((B,), num_vertices, edge_embedding_size)
            vertex_prop_input         = torch.cat([vertex_embedding, receiver_edge_embeddings], dim=dim_concat_vertex)
            vertex_embedding          = self.vertex_propagator(vertex_prop_input, res=step0_vertex_embedding) # ((B,), num_vertices, vertex_embedding_size)
        
        # decode vertex embedding back to 3D position
        delta_pos = self.decoder(vertex_embedding) # ((B,), num_vertices, 3)
        clamp_delta_pos = torch.clamp(delta_pos, min=-self.clamp_velocity, max=self.clamp_velocity)
        next_pos = latest_state + clamp_delta_pos
        return next_pos