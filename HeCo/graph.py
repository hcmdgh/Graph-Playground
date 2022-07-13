from dl import * 


class HeCoGraph:
    def __init__(self, 
                 hg: dgl.DGLHeteroGraph,
                 infer_node_type: str, 
                 metapath_list: list[list[str]],
                 relation_list: list[str],
                 sample_neighbor_cnt_list: list[int],
                 positive_sample: SparseTensor):
        self.hg = hg 
        self.metapath_list = metapath_list
        self.relation_list = relation_list
        self.sample_neighbor_cnt_list = sample_neighbor_cnt_list
        self.positive_sample = positive_sample 
        self.infer_node_type = infer_node_type
        assert len(relation_list) == len(sample_neighbor_cnt_list)
        
        self.feat_dict = dict(hg.ndata['feat']) 
        
        self.metapath_subgraph_list = [] 
        
        for metapath in metapath_list:
            subgraph = dgl.metapath_reachable_graph(g=hg, metapath=metapath)
            
            # subgraph = dgl.remove_self_loop(subgraph)
            # subgraph = dgl.add_self_loop(subgraph)
            
            self.metapath_subgraph_list.append(subgraph)
            
        self.relation_subgraph_list = [] 
        
        for relation in relation_list:
            subgraph = hg[relation]
            
            # subgraph = dgl.remove_self_loop(subgraph)
            # subgraph = dgl.add_self_loop(subgraph)
            
            self.relation_subgraph_list.append(subgraph)
