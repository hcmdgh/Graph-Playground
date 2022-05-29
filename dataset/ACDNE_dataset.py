from util import * 
import scipy.io as sio 

_DATASET_ROOT = './dataset/bin/ACDNE_dataset'


def _load_mat_file(path: str) -> HomoGraph:
    graph_dict = sio.loadmat(path)
    
    feat = graph_dict['attrb']
    adj_mat = graph_dict['network']
    label = graph_dict['group']

    if not isinstance(feat, ndarray):
        feat = feat.toarray()
    if not isinstance(adj_mat, ndarray):
        adj_mat = adj_mat.toarray()
    if not isinstance(label, ndarray):
        label = label.toarray()
    
    feat = feat.astype(np.float32)
    adj_mat = adj_mat.astype(np.int64)
    label = label.astype(np.int64)
    
    src_index, dest_index = np.argwhere(adj_mat > 0).T

    return HomoGraph(
        num_nodes = len(feat),
        edge_index = (torch.from_numpy(src_index), torch.from_numpy(dest_index)),
        node_prop_dict = {
            'feat': torch.from_numpy(feat),
            'label': torch.from_numpy(label), 
        },
        edge_prop_dict = {},
        num_classes = label.shape[-1],
    )


def load_ACDNE_dataset(dataset_name: Literal['acmv9', 'dblpv7', 'citationv1']) -> HomoGraph:
    pkl_path = os.path.join(_DATASET_ROOT, f'{dataset_name}.pt')
    
    if not os.path.isfile(pkl_path):
        homo_graph = _load_mat_file(os.path.join(_DATASET_ROOT, f'{dataset_name}.mat'))
        homo_graph.save_to_file(pkl_path)
    else:
        homo_graph = HomoGraph.load_from_file(pkl_path)
    
    return homo_graph
