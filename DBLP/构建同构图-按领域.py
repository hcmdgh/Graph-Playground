from conn import * 
import sys 
sys.path.append('..')
from util import * 


def main(
    field_names: list[str],
    max_cnt_per_field: int = 2000, 
    year_range: tuple[int, int] = (-1, 9999),
):
    init_log()
    device = auto_set_device()
    
    field2papers: dict[str, list[dict]] = defaultdict(list)
    
    for field_name in field_names:
        other_field_name_set = set(field_names)
        other_field_name_set.remove(field_name)
        
        resp = requests.get(
            url = f"{ES_HOST}/dblp_v11_paper/_search",
            auth = ES_AUTH,
            json = {
                'query': {
                    'bool': {
                        'must': [
                            {
                                'match': {
                                    'fos.name': field_name, 
                                }
                            },
                            {
                                'range': {
                                    'year': { 'gte': year_range[0], 'lte': year_range[1] }
                                }
                            },
                        ]
                    }
                },
                'sort': [{ 'n_citation': { 'order': 'desc' } }],
                'size': max_cnt_per_field, 
            },
        )
        
        resp_json = resp.json() 
        
        assert resp.status_code in range(200, 300) 

        for entry in resp_json['hits']['hits']:
            paper_entry = entry['_source']
            
            # [BEGIN] 过滤交叉领域的论文
            field_name_set = set(fos['name'] for fos in paper_entry['fos'])

            if field_name_set & other_field_name_set:
                continue
            # [END]
            
            field2papers[field_name].append(paper_entry)
            
    print("实际抽取论文数量统计：")
    for field_name, paper_list in field2papers.items():
        print(field_name, len(paper_list))
    print() 
    
    # [BEGIN] 构建同构图
    paper_id_map: dict[str, int] = dict() 
    
    for field_name, paper_list in field2papers.items():
        for paper_entry in paper_list:
            paper_id = paper_entry['id']
            
            if paper_id not in paper_id_map:
                paper_id_map[paper_id] = len(paper_id_map)
                
    edge_list = [] 
    
    for field_name, paper_list in field2papers.items():
        for paper_entry in paper_list:
            paper_id = paper_entry['id']
            ref_ids = paper_entry.get('references', [])
            
            for ref_id in ref_ids:
                if ref_id not in paper_id_map:
                    continue
                
                paper_nid = paper_id_map[paper_id]
                ref_nid = paper_id_map[ref_id]
                
                edge_list.append((paper_nid, ref_nid))
                edge_list.append((ref_nid, paper_nid))
                
    num_nodes = len(paper_id_map)

    edge_index = tuple(torch.tensor(edge_list, dtype=torch.int64).T)
    
    g = dgl.graph(edge_index, num_nodes=num_nodes)
    
    print(g)
    # [END]
    
    # [BEGIN] 生成结点特征
    year_th = torch.zeros(num_nodes, dtype=torch.int64)
    label_th = torch.zeros(num_nodes, dtype=torch.int64)
    title_list = [''] * num_nodes 
    
    for label, (field_name, paper_list) in enumerate(field2papers.items()):
        for paper_entry in paper_list:
            paper_id = paper_entry['id']
            paper_nid = paper_id_map[paper_id]
            paper_year = paper_entry['year']
            
            year_th[paper_nid] = paper_year
            label_th[paper_nid] = label 
            title_list[paper_nid] = paper_entry['title']
            
    emb = sbert_embedding(title_list)
    emb = perform_PCA(feat=emb, out_dim=64)
    
    emb_th = torch.tensor(emb, dtype=torch.float32)
    
    g.ndata['year'] = year_th 
    g.ndata['label'] = label_th 
    g.ndata['feat'] = emb_th 
    # [END]
    
    print(g)
    
    dgl.save_graphs('./DBLP/output/dblp_part.dgl', [g])


if __name__ == '__main__':
    main(
        field_names = [
            'Computer vision', 
            'Computer network', 
            'Data mining', 
            'Natural language processing', 
            'Software engineering',
            'Computer security',
            'Computer hardware',
        ],
        year_range = (2015, 2020),
        max_cnt_per_field = 2000, 
    ) 
