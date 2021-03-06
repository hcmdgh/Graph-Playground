from util import * 


class HGTConv(nn.Module):
    def __init__(self,
                 in_dim: Union[int, dict[str, int]],
                 out_dim: int,
                 node_types: set[str],
                 edge_types: set[tuple[str, str, str]],
                 num_heads: int = 1,):
        super().__init__()
        
        self.out_dim = out_dim 
        self.node_types = node_types 
        self.edge_types = edge_types
        self.num_heads = num_heads
        
        if isinstance(in_dim, int):
            self.in_dim_dict = {node_type: in_dim for node_type in node_types}
        elif isinstance(in_dim, dict):
            assert set(in_dim.keys()) == node_types 
            self.in_dim_dict = in_dim 
        else:
            raise AssertionError 
        
        self.K_fc_dict = nn.ModuleDict()
        self.Q_fc_dict = nn.ModuleDict()
        self.V_fc_dict = nn.ModuleDict()
        self.A_fc_dict = nn.ModuleDict()
        self.skip_alpha_dict = nn.ParameterDict()

        for node_type, in_dim in self.in_dim_dict.items():
            self.K_fc_dict[node_type] = nn.Linear(in_dim, out_dim)
            self.Q_fc_dict[node_type] = nn.Linear(in_dim, out_dim)
            self.V_fc_dict[node_type] = nn.Linear(in_dim, out_dim)
            self.A_fc_dict[node_type] = nn.Linear(out_dim, out_dim)
            self.skip_alpha_dict[node_type] = Parameter(torch.ones(1))

        self.A_rel_dict = nn.ParameterDict()
        self.M_rel_dict = nn.ParameterDict()
        self.P_rel_dict = nn.ParameterDict()

        assert out_dim % num_heads == 0 
        self.head_dim = out_dim // num_heads 
        
        for edge_type in edge_types:
            edge_type = '__'.join(edge_type)
            
            self.A_rel_dict[edge_type] = Parameter(torch.zeros(num_heads, self.head_dim, self.head_dim))
            self.M_rel_dict[edge_type] = Parameter(torch.zeros(num_heads, self.head_dim, self.head_dim))
            self.P_rel_dict[edge_type] = Parameter(torch.zeros(num_heads))

        self.reset_parameters() 
        
    def reset_parameters(self):
        deep_reset_parameters(self.K_fc_dict)
        deep_reset_parameters(self.Q_fc_dict)
        deep_reset_parameters(self.V_fc_dict)
        deep_reset_parameters(self.A_fc_dict)
        
        ones_(self.skip_alpha_dict)
        ones_(self.P_rel_dict)
        
        glorot_(self.A_rel_dict)
        glorot_(self.M_rel_dict)
        
    def forward(self,
                hg: dgl.DGLHeteroGraph,
                feat_dict: dict[str, FloatTensor]) -> dict[str, FloatTensor]:
        K_dict = dict()
        Q_dict = dict()
        V_dict = dict()
        _out_dict: dict[str, list[FloatTensor]] = dict()
        out_dict: dict[str, FloatTensor] = dict() 
        
        # [BEGIN] Step 1
        # ??????Q???K???V?????????
        # ??????????????????????????????????????????Q???K???V???????????????????????????????????????????????????
        for node_type, feat in feat_dict.items():
            # K_dict[i]: [num_nodes[i] x num_heads x head_dim]
            K_dict[node_type] = self.K_fc_dict[node_type](feat).view(-1, self.num_heads, self.head_dim)
            
            # Q_dict[i]: [num_nodes[i] x num_heads x head_dim]
            Q_dict[node_type] = self.Q_fc_dict[node_type](feat).view(-1, self.num_heads, self.head_dim)
            
            # V_dict[i]: [num_nodes[i] x num_heads x head_dim]
            V_dict[node_type] = self.V_fc_dict[node_type](feat).view(-1, self.num_heads, self.head_dim)

            _out_dict[node_type] = [] 
        # [END]
            
        for edge_type in self.edge_types:
            subgraph = hg[edge_type]
            
            src_type, _, dest_type = edge_type 
            edge_type = '__'.join(edge_type)
            
            # [BEGIN] Step 2
            # ???????????????Q???K???V?????????
            # K???V???????????????????????????????????????????????????????????????shape????????????
            # Q????????????????????????????????????
            # K???V????????????????????????????????????Q????????????????????????????????????
            
            # A_rel: [num_heads x head_dim x head_dim]
            A_rel = self.A_rel_dict[edge_type]
            
            # K_dict[src_type]: [num_nodes[src] x num_heads x head_dim] 
            # A_rel: [num_heads x head_dim x head_dim]
            # K: [num_nodes[src] x num_heads x head_dim] 
            K = (K_dict[src_type].transpose(0, 1) @ A_rel).transpose(1, 0)
            
            # M_rel: [num_heads x head_dim x head_dim]
            M_rel = self.M_rel_dict[edge_type]
            
            # V: [num_nodes[src] x num_heads x head_dim] 
            V = (V_dict[src_type].transpose(0, 1) @ M_rel).transpose(1, 0)

            # Q: [num_nodes[dest] x num_heads x head_dim] 
            Q = Q_dict[dest_type]
            # [END]
            
            # [BEGIN] Step 3
            # ????????????????????????????????????
            # K??????????????????Q???????????????????????????alpha????????????softmax???
            # V???????????????????????????alpha??????????????????????????????
            
            # P_rel: [num_heads]
            P_rel = self.P_rel_dict[edge_type]
            
            subgraph.srcdata['_K'] = K 
            subgraph.dstdata['_Q'] = Q 

            subgraph.apply_edges(dglfn.u_mul_v('_K', '_Q', '_alpha'))

            subgraph.srcdata.pop('_K')
            subgraph.dstdata.pop('_Q')
            
            # alpha: [num_edges x num_heads x head_dim]
            alpha = subgraph.edata.pop('_alpha')
            
            # alpha: [num_edges x num_heads]
            alpha = alpha.sum(dim=-1) * P_rel 
            alpha = alpha / math.sqrt(self.head_dim)
            alpha = dglF.edge_softmax(subgraph, alpha)
            
            # alpha: [num_edges x num_heads x 1]
            alpha = alpha.unsqueeze(dim=-1)
            subgraph.edata['_alpha'] = alpha 

            # V: [num_nodes[src] x num_heads x head_dim] 
            subgraph.srcdata['_V'] = V 
            
            subgraph.update_all(
                message_func = dglfn.u_mul_e('_V', '_alpha', '_'),
                reduce_func = dglfn.sum('_', '_out'),
            )
            
            subgraph.srcdata.pop('_V')
            
            # out: [num_nodes[dest] x num_heads x head_dim]
            out = subgraph.dstdata.pop('_out')
            
            # out: [num_nodes[dest] x out_dim]
            out = out.view(-1, self.out_dim)
            # [END]
            
            _out_dict[dest_type].append(out)

        # [BEGIN] Step 4 
        # ????????????????????????????????????
        # ??????????????????????????????????????????????????????????????????????????????
        # ?????????????????????????????????????????????????????????
        
        for node_type, out_list in _out_dict.items():
            # out: [num_nodes[i] x out_dim]
            out = torch.stack(out_list).mean(dim=0)
            assert out is not None 

            out = self.A_fc_dict[node_type](
                F.gelu(out)
            )
            
            # ??????????????????????????????????????????????????????
            if out.shape[-1] == feat_dict[node_type].shape[-1]:
                alpha = torch.sigmoid(
                    self.skip_alpha_dict[node_type]
                ) 
                
                out = alpha * out + (1 - alpha) * feat_dict[node_type]
                
            out_dict[node_type] = out                 
        # [END]

        return out_dict 
