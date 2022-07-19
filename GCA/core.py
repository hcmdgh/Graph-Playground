from model import * 
from util import * 

from dl import * 


@dataclass
class Config:
    graph: dgl.DGLGraph 
    
    emb_dim: int = 128
    drop_scheme: Literal['uniform', 'degree', 'pr', 'evc'] = 'degree'
    drop_edge_ratio_1: float = 0.6 
    drop_edge_ratio_2: float = 0.3 
    drop_feat_ratio_1: float = 0.2
    drop_feat_ratio_2: float = 0.3
    tau: float = 0.2 
    gnn_num_layers: int = 2 
    
    num_epochs: int = 1000 
    train_val_test_ratio: tuple[float, float, float] = (0.1, 0.45, 0.45)
    lr: float = 0.01 
    weight_decay: float = 1e-5 
    
    
class GCA_Core:
    def __init__(self, config: Config):
        self.config = config 
        
        device = get_device()
        
        self.graph = config.graph 
        self.num_nodes = self.graph.num_nodes() 
        self.feat = self.graph.ndata.pop('feat').to(device)
        self.feat_dim = self.feat.shape[-1]
        self.label = self.graph.ndata.pop('label').to(device)
        
        # [BEGIN] 转换为无向图并增加自环
        self.graph = dgl.to_bidirected(self.graph)
        self.graph = dgl.remove_self_loop(self.graph)
        self.graph = dgl.add_self_loop(self.graph)
        self.graph = self.graph.to(device)
        # [END]

        # [BEGIN] 计算边和结点特征的drop权重
        if config.drop_scheme == 'degree':
            self.edge_drop_weight = calc_edge_drop_weight(self.graph)
            self.feat_drop_weight = calc_feat_drop_weight(g=self.graph, feat=self.feat)
        elif config.drop_scheme == 'pr':
            raise NotImplementedError 
            drop_weights = pr_drop_weights(pyg_edge_index, aggr='sink', k=200).to(device)
        elif config.drop_scheme == 'evc':
            raise NotImplementedError
            drop_weights = evc_drop_weights(data).to(device)
        elif config.drop_scheme == 'uniform':
            self.edge_drop_weight = None
            self.feat_drop_weight = None 
        else:
            raise AssertionError
        # [END]

        self.model = GCA(
            in_dim = self.feat_dim, 
            emb_dim = config.emb_dim, 
            gnn_num_layers = config.gnn_num_layers, 
            tau = config.tau,
        )
        
        
    def drop_feat(self, drop_ratio: float) -> FloatTensor:
        if self.config.drop_scheme == 'uniform':
            return drop_feat(feat=self.feat, p=drop_ratio)
        elif self.config.drop_scheme in ['pr', 'degree', 'evc']:
            return drop_feat_weighted(
                feat = self.feat, 
                feat_weight = self.feat_drop_weight, 
                p = drop_ratio,
                threshold = 0.7, 
            )
        else:
            raise AssertionError 
        
        
    def drop_edge(self, drop_ratio: float) -> dgl.DGLGraph:
        if self.config.drop_scheme == 'uniform':
            return drop_edge(g=self.graph, p=drop_ratio)
        elif self.config.drop_scheme in ['degree', 'evc', 'pr']:
            return drop_edge_weighted(
                g = self.graph, 
                edge_weight = self.edge_drop_weight, 
                p = drop_ratio, 
                threshold = 0.7,
            )
        else:
            raise AssertionError
        
        
    def calc_semi_loss(self, 
                       h1: FloatTensor, 
                       h2: FloatTensor) -> FloatScalarTensor:
        exp = lambda x: torch.exp(x / self.config.tau)
        
        intra_sim = exp(calc_cosine_similarity(h1, h1))
        inter_sim = exp(calc_cosine_similarity(h1, h2))

        loss = -torch.log(
            inter_sim.diag() / 
            (inter_sim.diag() + (inter_sim.sum(dim=-1) - inter_sim.diag()) + (intra_sim.sum(dim=-1) - intra_sim.diag()))
        )

        return loss 
        
        
    def calc_loss(self, 
                  h1: FloatTensor, 
                  h2: FloatTensor) -> FloatScalarTensor:
        h1 = self.model.proj(h1)
        h2 = self.model.proj(h2)

        l1 = self.calc_semi_loss(h1, h2)
        l2 = self.calc_semi_loss(h2, h1)

        loss = torch.mean((l1 + l2) / 2) 

        return loss 
        
        
    def train_epoch(self) -> FloatScalarTensor:
        self.model.train() 
        
        corrupt_graph_1 = self.drop_edge(self.config.drop_edge_ratio_1)
        corrupt_graph_2 = self.drop_edge(self.config.drop_edge_ratio_2)

        corrupt_feat_1 = self.drop_feat(self.config.drop_feat_ratio_1)
        corrupt_feat_2 = self.drop_feat(self.config.drop_feat_ratio_2)

        emb_1 = self.model(g=corrupt_graph_1, feat=corrupt_feat_1)
        emb_2 = self.model(g=corrupt_graph_2, feat=corrupt_feat_2)

        loss = self.calc_loss(emb_1, emb_2)
        
        return loss 
    
    
    def eval_epoch(self,
                   label: FloatArrayTensor,
                   train_mask: BoolArrayTensor,
                   val_mask: BoolArrayTensor,
                   test_mask: BoolArrayTensor) -> dict[str, float]:
        self.model.eval() 
        
        with torch.no_grad():
            emb = self.model(g=self.graph, feat=self.feat).detach()

        eval_res = mlp_multiclass_classification(
            feat = emb, 
            label = label,
            train_mask = train_mask,
            val_mask = val_mask,
            test_mask = test_mask, 
        )

        return eval_res 
