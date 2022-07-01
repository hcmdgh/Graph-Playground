from util import * 
from .conn import * 

BERT_BATCH_SIZE = 64 


def main():
    device = auto_set_device()
    
    with open('./MAG/output/paper_detail_by_conference_filtered.pkl', 'rb') as fp:
        _paper_detail_dict = pickle.load(fp)

    paper_detail_dict: dict[int, tuple[str, list]] = dict() 
    
    for conference_name in _paper_detail_dict:
        for paper_id, paper_entry in _paper_detail_dict[conference_name].items():
            paper_detail_dict[paper_id] = (conference_name, paper_entry)
        
    if not os.path.isfile('./MAG/output/adj_list.pkl'):
        raise NotImplementedError
        conn, cursor = get_conn_cursor()
            
        adj_list: dict[int, set[int]] = defaultdict(set)
            
        for cname in tqdm(paper_detail_dict):
            for paper_id, paper_entry in tqdm(paper_detail_dict[cname].items()):
                cursor.execute(
                    "SELECT * FROM mag.paper_reference WHERE paper_id = %s",
                    [paper_id],
                )
                
                for entry in cursor.fetchall():
                    cited_paper_id = entry['paper_reference_id']
                    
                    if cited_paper_id in paper_id_set:
                        adj_list[paper_id].add(cited_paper_id)
                        adj_list[cited_paper_id].add(paper_id)
                        
        with open('./MAG/output/adj_list.pkl', 'wb') as fp:
            pickle.dump(adj_list, fp)
    else:
        with open('./MAG/output/adj_list.pkl', 'rb') as fp:
            adj_list = pickle.load(fp)
                
    nid_map: dict[int, int] = dict() 
    edge_index = [] 
    
    for src_nid in adj_list:
        for dest_nid in adj_list[src_nid]:
            if src_nid not in nid_map:
                nid_map[src_nid] = len(nid_map)
            if dest_nid not in nid_map:
                nid_map[dest_nid] = len(nid_map)    

            _src_nid = nid_map[src_nid]
            _dest_nid = nid_map[dest_nid]

            edge_index.append((_src_nid, _dest_nid))
            
    edge_index = np.array(edge_index).T 
    
    num_nodes = len(nid_map)
    print(f"num_nodes: {num_nodes}")
    print(f"num_edges: {edge_index.shape[1]}")
        
    # [BEGIN] 生成结点特征(BERT)
    node_emb_list = []
    
    nid_map_list = sorted(nid_map.items(), key=lambda x: x[1])
    assert [x[1] for x in nid_map_list] == list(range(num_nodes))
    
    for i in tqdm(range(0, num_nodes, BERT_BATCH_SIZE), desc='BERT'):
        text_batch = [paper_detail_dict[raw_pid][1][5] for raw_pid, new_pid in nid_map_list[i: i + BERT_BATCH_SIZE]]

        bert_emb = bert_embedding(text_batch)
        
        node_emb_list.append(bert_emb)
        
    node_emb = np.concatenate(node_emb_list, axis=0)
    
    assert node_emb.shape == (num_nodes, 768)
    
    with open('./MAG/output/node_emb.pkl', 'wb') as fp:
        pickle.dump(node_emb, fp)
    # [END]
    
    # [BEGIN] 生成结点标签
    cname_2_lid: dict[str, int] = dict()

    for cname in _paper_detail_dict:
        cname_2_lid[cname] = len(cname_2_lid)

    label_list: list[int] = [] 
        
    for raw_pid, new_pid in nid_map_list:
        cname = paper_detail_dict[raw_pid][0]
        
        label_list.append(cname_2_lid[cname])
        
    assert len(label_list) == num_nodes
    
    with open('./MAG/output/node_label.pkl', 'wb') as fp:
        pickle.dump(label_list, fp)
    # [END]

    # [BEGIN] 生成结点年份特征
    year_list = []

    for raw_pid, new_pid in nid_map_list:
        year = int(paper_detail_dict[raw_pid][1][7]) 
        
        year_list.append(year)
    # [END]
    
    # [BEGIN] 保存同构图
    g = dgl.graph(
        data = tuple(torch.from_numpy(edge_index)),
        num_nodes = num_nodes, 
    )
    
    g.ndata['label'] = torch.tensor(label_list, dtype=torch.int64)
    g.ndata['year'] = torch.tensor(year_list, dtype=torch.int64)
    g.ndata['feat'] = torch.tensor(node_emb, dtype=torch.float32)
    
    with open('./MAG/output/g.pkl', 'wb') as fp:
        pickle.dump(g, fp)
    # [END]
