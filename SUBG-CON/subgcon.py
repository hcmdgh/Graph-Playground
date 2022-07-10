from utils_mp import * 

import random
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from model import * 

EPS = 1e-15


class SUBG_CON(torch.nn.Module):
    def __init__(self,
                 in_dim: int,
                 emb_dim: int):
        super().__init__()
        
        self.gcn_encoder = GCNEncoder(in_dim=in_dim, out_dim=emb_dim)
        
        self.pool = dglnn.AvgPooling()
        
        self.ranking_loss = nn.MarginRankingLoss(0.5)

        self.reset_parameters()
        
        self.device = get_device()
        self.to(self.device)
        
    def reset_parameters(self):
        # TODO 
        pass 
        
    def forward(self,
                subgraph_batch: dgl.DGLGraph,
                center_nids: IntArray):
        feat = subgraph_batch.ndata['feat']
        h = self.gcn_encoder(subgraph_batch, feat)
        
        emb = h[center_nids]
        pooled_emb = self.pool(subgraph_batch, h)

        return emb, pooled_emb
    
    def calc_loss(self, 
                  emb: FloatTensor, 
                  pooled_emb: FloatTensor) -> FloatScalarTensor:
        N = len(emb)
        assert len(emb) == len(pooled_emb)
                  
        shuffled_idxs = torch.randperm(N)

        shuffled_emb = emb[shuffled_idxs]
        shuffled_pooled_emb = pooled_emb[shuffled_idxs]
        
        score_e_pe = torch.sigmoid(torch.sum(emb * pooled_emb, dim=-1))
        score_se_spe = torch.sigmoid(torch.sum(shuffled_emb * shuffled_pooled_emb, dim=-1))
        score_e_spe = torch.sigmoid(torch.sum(emb * shuffled_pooled_emb, dim=-1))
        score_se_pe = torch.sigmoid(torch.sum(shuffled_emb * pooled_emb, dim=-1))
        
        ones = torch.ones(N, device=self.device)
        
        loss = self.ranking_loss(score_e_pe, score_se_pe, ones) + self.ranking_loss(score_se_spe, score_e_spe, ones)
        # loss = self.ranking_loss(score_e_pe, score_e_spe, ones)
        
        return loss 
    
    def calc_node_emb(self, 
                      subgraph_service: SubgraphService,
                      full_num_nodes: int,
                      batch_size: int) -> FloatTensor:
        self.eval() 
        
        emb_list = [] 
        
        with torch.no_grad():
            for i in range(0, full_num_nodes, batch_size):
                nids = range(full_num_nodes)[i: i + batch_size]
                subgraph_batch, center_nids = subgraph_service.extract_subgraph_batch(nids)

                emb, _ = self(subgraph_batch=subgraph_batch, center_nids=center_nids)
                emb_list.append(emb)
        
        out = torch.cat(emb_list, dim=0)
        
        return out 
    
    def eval_graph(self,
                   subgraph_service: SubgraphService,
                   num_nodes: int,
                   batch_size: int,
                   label: IntTensor,
                   train_mask: BoolTensor,
                   val_mask: BoolTensor,
                   test_mask: BoolTensor) -> dict[str, float]:
        self.eval() 
        
        emb = self.calc_node_emb(
            subgraph_service = subgraph_service,
            full_num_nodes = num_nodes,
            batch_size = batch_size, 
        )
        
        clf_res = sklearn_multiclass_classification(
            feat = emb,
            label = label,
            train_mask = train_mask,
            val_mask = val_mask,
            test_mask = test_mask,
            max_epochs = 400, 
        )
        
        return clf_res
