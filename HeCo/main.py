from util import * 
from graph import * 
from model import * 

from dl import * 
import numpy
import torch
from utils import load_data, set_params, evaluate
import datetime
import pickle as pkl
import os
import random

args = set_params()
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

## name of intermediate document ##
own_str = args.dataset

## random seed ##
seed = args.seed
numpy.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def load_dataset(dataset_name: str) -> HeCoGraph:
    if dataset_name == 'ACM':
        hg = load_dgl_graph('/home/Dataset/GengHao/HeCo/ACM.pt')
        
        hg.nodes['author'].data['feat'] = torch.eye(hg.num_nodes('author'), dtype=torch.float32)
        hg.nodes['subject'].data['feat'] = torch.eye(hg.num_nodes('subject'), dtype=torch.float32)

        hg.nodes['paper'].data['feat'] = normalize_feature(hg.nodes['paper'].data['feat'])
        hg.nodes['author'].data['feat'] = normalize_feature(hg.nodes['author'].data['feat'])
        hg.nodes['subject'].data['feat'] = normalize_feature(hg.nodes['subject'].data['feat'])
        
        hg = hg.to(get_device())

        paper_positive_sample = sp.load_npz('/home/Dataset/HeCo/ACM/positive_sample_paper.npz')
        paper_positive_sample = to_torch_sparse(paper_positive_sample)
        
        metapath_list = [['pa', 'ap'], ['ps', 'sp']]
        relation_list = ['pa', 'ps']
        sample_neighbor_cnt_list = [7, 1]

        graph = HeCoGraph(
            hg = hg,
            infer_node_type = 'paper', 
            metapath_list = metapath_list,
            relation_list = relation_list,
            sample_neighbor_cnt_list = sample_neighbor_cnt_list, 
            positive_sample = paper_positive_sample, 
        )
        
        return graph 
    else:
        raise AssertionError 


@dataclass 
class Pipeline:
    dataset_name: Literal['ACM'] = 'ACM'
    
    def run(self):
        set_cwd(__file__)
        init_log()
        device = auto_set_device()
        
        wandb.init(project='HeCo', config=asdict(self))
        
        graph = load_dataset(self.dataset_name)
        
        infer_ntype = graph.infer_node_type 
        train_mask_list = [
            graph.hg.nodes[infer_ntype].data['train_mask_20'],
            graph.hg.nodes[infer_ntype].data['train_mask_40'],
            graph.hg.nodes[infer_ntype].data['train_mask_60'],
        ]
        val_mask_list = [
            graph.hg.nodes[infer_ntype].data['val_mask_20'],
            graph.hg.nodes[infer_ntype].data['val_mask_40'],
            graph.hg.nodes[infer_ntype].data['val_mask_60'],
        ]
        test_mask_list = [
            graph.hg.nodes[infer_ntype].data['test_mask_20'],
            graph.hg.nodes[infer_ntype].data['test_mask_40'],
            graph.hg.nodes[infer_ntype].data['test_mask_60'],
        ]
        
        label = graph.hg.nodes[infer_ntype].data['label']
        num_classes = int(torch.max(label)) + 1 
        
        model = HeCo(
            graph = graph, 
            emb_dim = args.hidden_dim, 
            feat_dropout = args.feat_drop, 
            attn_dropout = args.attn_drop,
            tau = args.tau, 
            lam = args.lam,
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)

        cnt_wait = 0
        best = 1e9
        best_t = 0

        for epoch in range(1, args.nb_epochs + 1):
            model.train()

            loss = model()

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step() 
            
            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                # torch.save(model.state_dict(), 'HeCo_'+own_str+'.pkl')
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                print('Early stopping!')
                break
            
            if epoch % 10 == 0:
                clf_res = model.eval_graph()

                log_multi(
                    wandb_log = True, 
                    epoch = epoch,
                    loss = float(loss), 
                    **clf_res,
                ) 
            else:
                log_multi(
                    wandb_log = True, 
                    epoch = epoch,
                    loss = float(loss), 
                ) 
            

if __name__ == '__main__':
    pipeline = Pipeline(
        dataset_name = 'ACM',
    )
    
    pipeline.run() 
