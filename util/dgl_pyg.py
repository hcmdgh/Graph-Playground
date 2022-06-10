from .imports import * 


def get_dgl_hg_edge_index_dict(hg: dgl.DGLHeteroGraph,
                               return_type: Literal['dgl', 'pyg']) -> Union[dict[tuple[str, str, str], tuple[FloatTensor, FloatTensor]], dict[tuple[str, str, str], FloatTensor]]:
    edge_index_dict = dict() 
    
    for edge_type in hg.canonical_etypes:
        if return_type == 'dgl':
            edge_index = hg.edges(etype=edge_type)
        elif return_type == 'pyg':
            edge_index = torch.stack(hg.edges(etype=edge_type)) 
        else:
            raise AssertionError 
        
        edge_index_dict[edge_type] = edge_index 
        
    return edge_index_dict
