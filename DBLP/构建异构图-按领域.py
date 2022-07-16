from conn import * 
import sys 

from dl import * 


def main(
    field_names: list[str],
    max_cnt_per_field: int = 2000, 
    year_range: tuple[int, int] = (-1, 9999),
):
    set_cwd(__file__)
    init_log()
    device = auto_set_device()
    
    field2papers: dict[str, list[dict]] = defaultdict(list)
    
    for field_name in tqdm(field_names):
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
    
    print("按年份统计论文数量：")
    year2cnt = defaultdict(int)
    for field_name, paper_list in field2papers.items():
        for paper_entry in paper_list:
            year = paper_entry['year']
            year2cnt[year] += 1 
    pprint(year2cnt)
    print() 
    
    # [BEGIN] 构建异构图
    paper_id_map: dict[str, int] = dict() 
    author_id_map: dict[str, int] = dict() 
    venue_id_map: dict[str, int] = dict() 
    
    for field_name, paper_list in field2papers.items():
        for paper_entry in paper_list:
            paper_id = paper_entry['id']
            
            if paper_id not in paper_id_map:
                paper_id_map[paper_id] = len(paper_id_map)
                
            if paper_entry.get('authors'):
                author_id_list = [author_entry['id'] for author_entry in paper_entry['authors']]

                for author_id in author_id_list:
                    if author_id not in author_id_map:
                        author_id_map[author_id] = len(author_id_map)

            if paper_entry.get('venue'):
                venue = paper_entry['venue']['raw']
                
                if venue not in venue_id_map:
                    venue_id_map[venue] = len(venue_id_map)
                
    pp_edge_list = [] 
    pa_edge_list = [] 
    pv_edge_list = [] 
    
    for field_name, paper_list in field2papers.items():
        for paper_entry in paper_list:
            paper_id = paper_entry['id']
            paper_nid = paper_id_map[paper_id]
            ref_ids = paper_entry.get('references', [])
            
            for ref_id in ref_ids:
                if ref_id not in paper_id_map:
                    continue
                
                ref_nid = paper_id_map[ref_id]
                
                pp_edge_list.append((paper_nid, ref_nid))
                pp_edge_list.append((ref_nid, paper_nid))

            if paper_entry.get('authors'):
                author_id_list = [author_entry['id'] for author_entry in paper_entry['authors']]

                for author_id in author_id_list:
                    pa_edge_list.append((paper_nid, author_id_map[author_id]))

            if paper_entry.get('venue'):
                venue = paper_entry['venue']['raw']
                
                pv_edge_list.append((paper_nid, venue_id_map[venue]))
                
    num_nodes = len(paper_id_map)

    pp_edge_index = tuple(torch.tensor(pp_edge_list, dtype=torch.int64).T)
    pa_edge_index = tuple(torch.tensor(pa_edge_list, dtype=torch.int64).T)
    pv_edge_index = tuple(torch.tensor(pv_edge_list, dtype=torch.int64).T)
    
    hg = dgl.heterograph({
        ('paper', 'pp', 'paper'): pp_edge_index,
        ('paper', 'pa', 'author'): pa_edge_index,
        ('paper', 'pv', 'venue'): pv_edge_index,
        ('author', 'ap', 'paper'): pa_edge_index[::-1],
        ('venue', 'vp', 'paper'): pv_edge_index[::-1],
    })
    
    print("异构图信息：")
    print(hg)
    print() 
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
    
    hg.nodes['paper'].data['year'] = year_th 
    hg.nodes['paper'].data['label'] = label_th 
    hg.nodes['paper'].data['feat'] = emb_th 
    # [END]
    
    print(hg)
    
    save_dgl_graph(hg, f'./output/dblp_{year_range[0]}_{year_range[1]}.pt')


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
        year_range = (2014, 2020),
        max_cnt_per_field = 2000, 
    ) 
