from util import * 
dglnn.GraphConv
dglnn.SAGEConv
pygnn.SAGEConv


class MyGCN(pygnn.MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')
        
    def forward(self,
                edge_index_dict,
                feat_dict):
        out = self.propagate(
            torch.stack(edge_index_dict[('paper', 'pa', 'author')]),
            src_feat = feat_dict['paper'], 
            dest_feat = feat_dict['author'],
        )
        
        return out 
    
    def message(self,
                src_feat_j,
                dest_feat_i):
        out = src_feat_j * 2 + dest_feat_i
        
        return out 

        
def msg_func(edges):
    src = edges.src['feat']
    dest = edges.dst['feat'] 
    
    return { 'm': src * 2 + dest }


def reduce_func(nodes):
    mailbox = nodes.mailbox 
    
    return { 'h': torch.sum(mailbox['m'], dim=1) }


hg_graph = HeteroGraph.generate_bipartite()

hg = hg_graph.to_dgl(with_attr=True)

gcn = MyGCN() 

out = gcn(hg_graph.edge_index_dict, hg_graph.node_attr_dict['feat'])

hg.update_all(message_func=msg_func,
              reduce_func=dglfn.sum('m', 'h'))

out2 = hg.nodes['author'].data['h']

print(out)
print(out2)