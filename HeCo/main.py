from util import * 
from graph import * 
from model import * 

from dl import * 


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
    attn_dropout: float = 0.5 
    feat_dropout: float = 0.3 
    emb_dim: int = 64 
    lam: float = 0.5 
    tau: float = 0.8 
    
    seed: Optional[int] = None 
    lr: float = 0.0008 
    weight_decay: float = 0. 
    
    def run(self):
        set_cwd(__file__)
        init_log()
        device = auto_set_device()
        seed_all(self.seed)
        
        wandb.init(project='HeCo', config=asdict(self))
        
        graph = load_dataset(self.dataset_name)
        
        model = HeCo(
            graph = graph, 
            emb_dim = self.emb_dim, 
            feat_dropout = self.feat_dropout, 
            attn_dropout = self.attn_dropout,
            tau = self.tau, 
            lam = self.lam,
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        cnt_wait = 0
        best = 1e9

        for epoch in itertools.count(1):
            model.train()

            loss = model()

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step() 
            
            if loss < best:
                best = loss
                cnt_wait = 0
                # torch.save(model.state_dict(), 'HeCo_'+own_str+'.pkl')
            else:
                cnt_wait += 1

            if cnt_wait == 10:
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
