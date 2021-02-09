import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TemporalLayer(nn.Module):
    
    def __init__(self, entity_index, input_units, units, activation):
        
        '''
        entity_index: list containing tensors, each of which 
        refers to the list of indexes of entities of one type
        
        input_units: length of the current embedding
        units: length of the historical hidden embedding
        activation: activation function for FC_F
        '''
        
        super(TemporalLayer, self).__init__()
        
        self.entity_indexs = entity_index
        self.num_entity = len(entity_index)
        self.input_units = input_units
        self.units = units
        
        self.sin_mask = torch.LongTensor(np.array([1, -1] * int(units/2))) > 0
        self.cos_mask = torch.LongTensor(np.array([1, -1] * int(units/2))) < 0
        
        self.q_w = nn.Parameter(
            torch.randn((self.num_entity, self.input_units, self.units), dtype=torch.float), requires_grad = True
        )
        
        self.k_w = nn.Parameter(
            torch.randn((self.num_entity, self.units, self.units), dtype=torch.float), requires_grad = True
        )
        
        self.v_w = nn.Parameter(
            torch.randn((self.num_entity, self.units, self.units), dtype=torch.float), requires_grad = True
        )
        
        self.fc_w= nn.Parameter(
            torch.randn((self.num_entity, self.units, self.units), dtype=torch.float), requires_grad = True
        )
        
        self.activation = activation
        
    def attribute_temporal_encoding(self, historical_embedding):
        
        def get_temporal(time, embedding_length):
            t_sin = torch.tensor([[time / (10000 ** (2 * x / embedding_length)) for x in range(embedding_length)]])
            t_cos = torch.tensor([[time / (10000 ** ((2 * x - 1) / embedding_length)) for x in range(embedding_length)]])
            sin_mask = torch.LongTensor(np.array([1, -1] * int(embedding_length/2))) > 0
            cos_mask = torch.LongTensor(np.array([1, -1] * int(embedding_length/2))) < 0
            sin = torch.sin(t_sin) * sin_mask
            cos = torch.cos(t_cos) * cos_mask
            return sin + cos
        
        embedding_length = historical_embedding.shape[-1]
        historical_length = historical_embedding.shape[-2]
        num_of_points = historical_embedding.shape[-3]
        
        temporal_attribute = torch.unsqueeze(
            torch.cat(
            [get_temporal(x, embedding_length) for x in range(historical_length)]
            ), 0).repeat(num_of_points, 1, 1)
        
        return historical_embedding + temporal_attribute
        
    def forward(self, current_node_state, historical_buffer):
        '''
        current_node_state: shape(n_points, 1, input_shape[1]) 
        historical_buffer: shape(n_points, history_length, self.units)
        '''
        node_state = []
        for type_index, entity_index in enumerate(self.entity_indexs):
            entity_index = torch.LongTensor(entity_index)
            this_q_w = self.q_w[type_index]
            this_k_w = self.k_w[type_index]
            this_v_w = self.v_w[type_index]
            this_fc_w = self.fc_w[type_index]
            this_center_embedding = current_node_state[entity_index].unsqueeze(1)
            this_historical_embedding = historical_buffer[entity_index]
            # this_historical_embedding_ = self.attribute_temporal_encoding(this_historical_embedding)
            q = torch.matmul(this_center_embedding, this_q_w)
            k = torch.matmul(this_historical_embedding, this_k_w)
            v = torch.matmul(this_historical_embedding, this_v_w)
            raw_score = torch.matmul(q, k.permute(0, 2, 1))
            attention_score = F.softmax(raw_score)
            agg_node_state = torch.squeeze(torch.matmul(attention_score, v), 0)
            agg_node_state = self.activation(torch.matmul(agg_node_state, this_fc_w))
            node_state.append(agg_node_state + q)
            
        return torch.squeeze(torch.cat(node_state))

class SpatialLayer(nn.Module):
    
    def __init__(self, input_units, units, relations, entities, activation):
        
        '''
        input_units: length of the node embedding
        units: length of the hidden embedding 
        relations: number of heterogeneous relations
        entities: number of heterogeneous entitiy types
        activation: activation function for FC_F
        '''
        
        super(SpatialLayer, self).__init__()
        
        self.input_units = input_units
        self.units = units
        self.relations = relations
        self.entities = entities
        self.activation = activation
                
        self.point_enc_w = nn.Parameter(
            torch.randn((self.entities, self.input_units, self.units), dtype=torch.float), requires_grad = True
        )
        
        self.relation_enc_w = nn.Parameter(
            torch.randn((self.relations, self.units, self.units), dtype=torch.float), requires_grad = True
        )
        
        self.q_w = nn.Parameter(
            torch.randn((self.units, self.units), dtype=torch.float), requires_grad = True
        )
        
        self.k_w = nn.Parameter(
            torch.randn((self.units, self.units), dtype=torch.float), requires_grad = True
        )
        
        self.v_w = nn.Parameter(
            torch.randn((self.units, self.units), dtype=torch.float), requires_grad = True
        )
        
        self.fc_f = nn.Linear(self.units, self.units)
        
    def forward(self, node_state, adjacency, point_enc, relation_enc):
        '''
        node_state: shape(n_points, input_shape[1]) 
        adjacency: shape(n_points, None) [[1,2,3,4,5], [11,22]......]
        point enc: shape(n_points, 1) [[1],[1],[1],[2],[2],[2]....]
        relation_enc: shape(n_points, None) [[1,2,3,4], [2,1,1]......]
        '''
        
        # address the node heterogeneity by attributing point encoding
        point_encoding = self.point_enc_w[point_enc]
        node_state = torch.matmul(node_state.unsqueeze(1), point_encoding)
        
        center_embedding = node_state

        # insert fictional neighbor placeholder 
        node_state = torch.cat([torch.tensor([[[0] * self.units]]), node_state])

        # get neighbors of each point
        neighbors = torch.matmul(node_state[adjacency], self.relation_enc_w[relation_enc])
        
        # q, k, v operation
        q = torch.matmul(center_embedding, self.q_w) / (self.units ** 0.5)
        k = torch.squeeze(torch.matmul(neighbors, self.k_w), -2)  / (self.units ** 0.5)
        v = torch.squeeze(torch.matmul(neighbors, self.v_w))
        # attention score calculation
        raw_score = torch.matmul(q, k.permute(0, 2, 1))
        raw_score[relation_enc.unsqueeze(1) == 0] = -1e9
        attention_score = F.softmax(raw_score)

        # neighbor aggregation
        agg_node_state = torch.matmul(attention_score, v)
        agg_node_state = center_embedding + self.activation(self.fc_f(agg_node_state))
        
        return torch.squeeze(agg_node_state)