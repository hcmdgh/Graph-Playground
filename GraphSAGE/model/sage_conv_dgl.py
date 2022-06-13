from util import * 


class SAGEConv(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 aggr_type: Literal['mean', 'gcn', 'pool', 'lstm'],
                 dropout: float = 0.0,
                 bias: bool = True,
                 norm: Optional[Callable] = None,
                 act: Optional[Callable] = None):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aggr_type = aggr_type 
        self.norm = norm 
        self.act = act 
        
        self.dropout = nn.Dropout(dropout)
        
        self.pool_fc = self.lstm = self.self_fc = None 
        
        if aggr_type == 'pool':
            self.pool_fc = nn.Linear(in_dim, in_dim)
        elif aggr_type == 'lstm':
            self.lstm = nn.LSTM(
                input_size = in_dim, 
                hidden_size = in_dim, 
                batch_first=True,
            )
        elif aggr_type == 'mean':
            pass
        elif aggr_type == 'gcn':
            pass 
        else:
            raise AssertionError 
        
        if aggr_type != 'gcn':
            self.self_fc = nn.Linear(in_dim, out_dim, bias=False)
        
        self.neigh_fc = nn.Linear(in_dim, out_dim, bias=False)
        
        if bias:
            self.bias = Parameter(torch.zeros(out_dim))
        else:
            self.bias = None 
            
        self.reset_parameters() 
        
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        
        if self.aggr_type == 'pool':
            nn.init.xavier_uniform_(self.pool_fc.weight, gain=gain)
        elif self.aggr_type == 'lstm':
            self.lstm.reset_parameters()
        elif self.aggr_type == 'mean':
            pass
        elif self.aggr_type == 'gcn':
            pass 
        else:
            raise AssertionError 
        
        if self.aggr_type != 'gcn':
            nn.init.xavier_uniform_(self.self_fc.weight, gain=gain)

        nn.init.xavier_uniform_(self.neigh_fc.weight, gain=gain)
    
    def forward(self,
                graph: dgl.DGLGraph,
                feat: FloatTensor,
                edge_weight: Optional[FloatTensor] = None) -> FloatTensor:
        # feat: float[num_nodes x in_dim]
                
        with graph.local_scope():
            # feat_drop: [num_nodes x in_dim]
            feat_drop = self.dropout(feat)
            
            if edge_weight is None:
                msg_fn = dglfn.copy_src('_feat', '_')
            else:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata['_edge_weight'] = edge_weight
                msg_fn = dglfn.u_mul_e('_feat', '_edge_weight', '_')
                
            feat_self = feat_drop
            
            assert graph.num_edges() > 0
            
            lin_before_msg_pass = self.in_dim > self.out_dim
            
            if self.aggr_type == 'mean':
                if lin_before_msg_pass:
                    # _feat: [num_nodes x out_dim]
                    graph.ndata['_feat'] = self.neigh_fc(feat_drop)
                else:
                    graph.ndata['_feat'] = feat_drop

                graph.update_all(message_func=msg_fn,
                                 reduce_func=dglfn.mean('_', '_neigh'))
                
                # feat_neigh: [num_nodes x out_dim]
                feat_neigh = graph.ndata.pop('_neigh') 
                
                if not lin_before_msg_pass:
                    # feat_neigh: [num_nodes x out_dim]
                    feat_neigh = self.neigh_fc(feat_neigh)
                    
            elif self.aggr_type == 'gcn':
                if lin_before_msg_pass:
                    # _feat: [num_nodes x out_dim]
                    graph.ndata['_feat'] = self.neigh_fc(feat_drop)
                else:
                    graph.ndata['_feat'] = feat_drop
                    
                graph.update_all(message_func=msg_fn, 
                                 reduce_func=dglfn.sum('_', '_neigh'))

                # [BEGIN] 除以入度
                in_degrees = graph.in_degrees().to(feat_drop)

                # feat_neigh: [num_nodes x out_dim]
                feat_neigh = (graph.ndata['_neigh'] + graph.ndata['_feat']) \
                             / (in_degrees.unsqueeze(-1) + 1)
                # [END]
                    
                if not lin_before_msg_pass:
                    # feat_neigh: [num_nodes x out_dim]
                    feat_neigh = self.neigh_fc(feat_neigh)
                    
            elif self.aggr_type == 'pool':
                # _feat: [num_nodes x in_dim]
                graph.ndata['_feat'] = torch.relu(
                    self.pool_fc(feat_drop)
                )
                
                graph.update_all(msg_fn=msg_fn,
                                 reduce_func=dglfn.max('_', '_neigh'))

                # feat_neigh: [num_nodes x out_dim]
                feat_neigh = self.neigh_fc(graph.ndata.pop('_neigh'))

            elif self.aggr_type == 'lstm':
                raise NotImplementedError  
            else:
                raise AssertionError 
            
            if self.aggr_type == 'gcn':
                feat_out = feat_neigh 
            else:
                feat_out = self.self_fc(feat_self) + feat_neigh 
                
            if self.bias is not None:
                feat_out = feat_out + self.bias 
                
            if self.act is not None:
                feat_out = self.act(feat_out)
                
            if self.norm is not None:
                feat_out = self.norm(feat_out)
                
            # feat_out: [num_nodes x out_dim]
            return feat_out