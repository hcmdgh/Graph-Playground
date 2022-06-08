from util import * 


def parse_mat_file(in_path: str, out_path: str):
    net = sio.loadmat(in_path)
    x, a, y = net['attrb'], net['network'], net['group']

    feat = torch.tensor(x, dtype=torch.float32)
    label = torch.tensor(y, dtype=torch.int64)
    num_nodes = len(feat)
    
    adj_mat = a.toarray()
    indices = np.argwhere(adj_mat > 0)
    src_index, dest_index = indices.T 
    src_index = torch.tensor(src_index, dtype=torch.int64)
    dest_index = torch.tensor(dest_index, dtype=torch.int64)

    graph = HomoGraph(
        num_nodes = num_nodes,
        edge_index = (src_index, dest_index),
        node_attr_dict = { 'feat': feat, 'label': label, },
        num_classes = label.shape[1], 
    )
    
    graph.save_to_file(out_path)
    

def main():
    parse_mat_file(in_path='/home/Dataset/HomoGraphTL/ACDNE/acmv9.mat', out_path='/home/Dataset/GengHao/HomoGraph/ACDNE/acmv9.pt') 
    parse_mat_file(in_path='/home/Dataset/HomoGraphTL/ACDNE/citationv1.mat', out_path='/home/Dataset/GengHao/HomoGraph/ACDNE/citationv1.pt') 
    parse_mat_file(in_path='/home/Dataset/HomoGraphTL/ACDNE/dblpv7.mat', out_path='/home/Dataset/GengHao/HomoGraph/ACDNE/dblpv7.pt') 


if __name__ == '__main__':
    main() 
