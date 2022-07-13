from util import * 
from graph import * 

from dl import * 


class InterAttention(nn.Module):
    def __init__(self, 
                 emb_dim: int, 
                 attn_dropout: float):
        super().__init__()

        self.fc = nn.Linear(emb_dim, emb_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = Parameter(torch.empty(size=(1, emb_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax()
        if attn_dropout:
            self.attn_drop = nn.Dropout(attn_dropout)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        z_mc = 0
        for i in range(len(embeds)):
            z_mc += embeds[i] * beta[i]
        return z_mc


class IntraAttention(nn.Module):
    def __init__(self, 
                 emb_dim: int, 
                 attn_dropout: float):
        super().__init__()

        self.att = Parameter(torch.empty(size=(1, 2*emb_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)
        if attn_dropout:
            self.attn_drop = nn.Dropout(attn_dropout)
        else:
            self.attn_drop = lambda x: x

        self.softmax = nn.Softmax(dim=1)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, nei, h, h_refer):
        # nei_emb = F.embedding(nei, h)
        nei_emb = h[nei]
        h_refer = torch.unsqueeze(h_refer, 1)
        h_refer = h_refer.expand_as(nei_emb)
        all_emb = torch.cat([h_refer, nei_emb], dim=-1)
        attn_curr = self.attn_drop(self.att)
        att = self.leakyrelu(all_emb.matmul(attn_curr.t()))
        att = self.softmax(att)
        nei_emb = (att*nei_emb).sum(dim=1)
        return nei_emb


class SchemaEncoder(nn.Module):
    def __init__(self, 
                 graph: HeCoGraph, 
                 emb_dim: int, 
                 attn_dropout: float):
        super().__init__()
        
        self.graph = graph 

        self.intra_attn_list = nn.ModuleList([
            IntraAttention(emb_dim=emb_dim, attn_dropout=attn_dropout) 
            for _ in range(len(graph.relation_list))
        ])
        
        self.inter_attn = InterAttention(emb_dim=emb_dim, attn_dropout=attn_dropout)

    def forward(self,
                feat_dict: dict[str, FloatTensor]) -> FloatTensor:
        emb_list = []
        
        for subgraph, intra_attn, sample_neighbor_cnt in zip(self.graph.relation_subgraph_list, self.intra_attn_list, self.graph.sample_neighbor_cnt_list):
            sampled_adj_arr = sample_neighbor_from_graph(
                graph = subgraph,
                sample_neighbor_cnt = sample_neighbor_cnt, 
            )
            sampled_adj_th = torch.from_numpy(sampled_adj_arr).to(torch.int64).to(get_device())
            
            src_ntype = subgraph.srctypes[0] 
            dest_ntype = subgraph.dsttypes[0] 
            src_feat = feat_dict[src_ntype]
            dest_feat = feat_dict[dest_ntype]
            
            emb = F.elu(intra_attn(sampled_adj_th, dest_feat, src_feat))
            emb_list.append(emb)
            
        z_mc = self.inter_attn(emb_list)

        return z_mc
