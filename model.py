import math
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F

class gcn_base_model(nn.Module):
    def __init__(self):
        super(gcn_base_model,self).__init__()

    def set_adj(self,adj):
        self.adj = adj

    def normalize_adj(self):
        normalized_adj = self.adj
        deg_matrix = normalized_adj.sum(dim=1).unsqueeze(1)
        sqrt_deg_matrix = (deg_matrix+1e-7).pow(-1)
        normalized_adj = sqrt_deg_matrix * normalized_adj
        self.norm_adj = normalized_adj
        return normalized_adj


class model_8_2(gcn_base_model):
    def __init__(self, num_heads, num_self_att, hidden1=32, keep_short=24,keep_long=21):
        super(model_8_2, self).__init__()

        # configs
        self.hidden1 = hidden1
        self.keep_short = keep_short
        self.keep_long = keep_long
        self.num_heads = num_heads
        self.num_self_att = num_self_att

        # embeddings
        self.hour_embedding = nn.Embedding(24,hidden1)
        self.day_embedding = nn.Embedding(2,hidden1)
        self.location_embedding = nn.Embedding(171,hidden1)


        self.self_att_projs = nn.ModuleList([
            nn.ModuleList([nn.Linear(hidden1,hidden1//num_heads) for i in range(num_heads)]) for j in range(self.num_self_att)])

        self.fc_short = nn.Linear(keep_short, hidden1)
        self.fc_long = nn.Linear(keep_long, hidden1)

        # gcns
        self.gcn_w1 = nn.Linear(hidden1*5,hidden1*5)
        self.gcn_w2 = nn.Linear(hidden1*5,hidden1)

        self.fc_1_layer=  nn.Linear(hidden1*6, hidden1*6)
        self.fc_2_layer = nn.Linear(hidden1*6,1)

        self.rainfall_gate = nn.Linear(1,1) 

    def perform_gcn(self, _input):
        # _input : batch_num x 171 x feature_len
        # gcn layer 1
        h_gcn = torch.matmul(self.norm_adj,_input.permute([1,0,2]).contiguous().view(171,-1)).view(171,-1,self.hidden1*5)
        h_gcn = F.dropout(F.relu(self.gcn_w1(h_gcn)), p=0.5, training=self.training)

        # gcn layer 2
        h_gcn = torch.matmul(self.norm_adj,h_gcn.view(171,-1)).view(171,-1,self.hidden1*5).permute([1,0,2])
        h_gcn = F.dropout(F.relu(self.gcn_w2(h_gcn)), p=0.5, training=self.training)

        return h_gcn

    def perform_multi_head_att(self, _input, projection_id):
        # _input : batch_num x 171 x seq_len x feature_len
        # multi-head attention
        batch_num = _input.size(0)
        seq_len = _input.size(2)
        item_list = []
        for i in range(self.num_heads):
            _items = self.self_att_projs[projection_id][i](_input.view(-1,seq_len, self.hidden1)) # batch_num*171 x seq_len x 8
            weights = F.softmax(_items.bmm(_items.permute([0,2,1]))/math.sqrt(32), dim= 2) # batch_num*171 x seq_len x 5
            _items = weights.bmm(_items) # batch_num*171 x seq_len x 8
            item_list.append(_items)

        h_att = torch.cat(item_list, dim=2).view(batch_num, 171, seq_len, 32) + _input
        return h_att


    def forward(self, x_hour_short, x_hour_long, hour_code, day_code, location_code, rainfall):
        batch_num = x_hour_short.size(0)

         # feature trimming
        x_hour_short = x_hour_short[:,:,-self.keep_short:]
        x_hour_long = x_hour_long[:,:,-self.keep_long:]
        rainfall = rainfall[:,:,-1:].repeat(1,171,1)

        #  batch_num x seq_len x hidden_dim
        h_short = self.fc_short(x_hour_short)
        h_long = self.fc_long(x_hour_long)

        hour_emb = self.hour_embedding(hour_code).unsqueeze(1).repeat(1,171,1)
        day_emb = self.day_embedding(day_code).unsqueeze(1).repeat(1,171,1)
        location_emb = self.location_embedding(location_code)

        items = torch.stack([h_short, h_long, hour_emb, day_emb, location_emb], dim=2) # batch_num x 171 x 5 x 32
        h_gcn = self.perform_gcn(items.view(batch_num, 171, -1))

        items = torch.cat([h_gcn.unsqueeze(2), items], dim=2) # batch_num x 171 x 6 x 32

        # multi-head attention
        h_att = items
        for i in range(self.num_self_att):
            h_att = self.perform_multi_head_att(h_att, projection_id=i)

        h_att = h_att.view(batch_num, 171, -1)
        out = F.dropout(F.relu(self.fc_1_layer(h_att)),p= 0.5, training=self.training)
        out = F.relu(self.fc_2_layer(out))

        gate = torch.sigmoid(self.rainfall_gate(rainfall))
        return (out*gate).squeeze(dim=2)





