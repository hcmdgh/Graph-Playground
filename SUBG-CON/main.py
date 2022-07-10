import argparse, os
import math
import torch
import random
import numpy as np
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

from utils_mp import *
from subgcon import SUBG_CON
from model import GCNEncoder, Pool
from dl import * 

    
@dataclass
class Pipeline:
    graph: dgl.DGLGraph
    batch_size: int = 500 
    neighbor_sample_cnt: int = 20 
    PPR_order: int = 10 
    emb_dim: int = 512
    
    def run(self):
        set_cwd(__file__)
        init_log()
        device = auto_set_device()
        
        graph = self.graph.to(device)
        num_nodes = graph.num_nodes()
        feat = graph.ndata['feat']
        feat_dim = feat.shape[-1]
        train_mask = graph.ndata.pop('train_mask')
        val_mask = graph.ndata.pop('val_mask')
        test_mask = graph.ndata.pop('test_mask')
        label = graph.ndata.pop('label')
        
        if not os.path.isfile('./output/subgraph_service.pkl'):
            subgraph_service = SubgraphService(graph=graph, neighbor_sample_cnt=self.neighbor_sample_cnt, PPR_order=self.PPR_order)
            subgraph_service.save_to_file('./output/subgraph_service.pkl')
        else:
            subgraph_service = SubgraphService.load_from_file('./output/subgraph_service.pkl')
        
        model = SUBG_CON(
            in_dim = feat_dim, 
            emb_dim = self.emb_dim, 
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
        def train(epoch):
            model.train()
            optimizer.zero_grad()
            sample_idx = random.sample(range(num_nodes), self.batch_size)
            subgraph_batch, center_nids = subgraph_service.extract_subgraph_batch(sample_idx)
            emb, pooled_emb = model(subgraph_batch=subgraph_batch, center_nids=center_nids)
            
            loss = model.calc_loss(emb, pooled_emb)
            loss.backward()
            optimizer.step()
            return loss.item()
        
        for epoch in range(120):
            loss = train(epoch)
            print('epoch = {}, loss = {}'.format(epoch, loss))
            # val_acc, test_acc = test(model) 
            
            eval_res = model.eval_graph(
                subgraph_service = subgraph_service,
                num_nodes = num_nodes,
                batch_size = self.batch_size,
                label = label,
                train_mask = train_mask,
                val_mask = val_mask,
                test_mask = test_mask, 
            )
            
            print(eval_res)


if __name__ == '__main__':
    pipeline = Pipeline(
        graph = load_dgl_dataset('cora'), 
    )
    
    pipeline.run() 
