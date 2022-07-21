from query import * 
import sys 
sys.path.append('..')
from util import * 

FIELD_NAMES = [
    'Computer vision', 
    'Computer network', 
    'Data mining', 
    'Natural language processing', 
    'Software engineering',
    'Computer security',
    'Computer hardware',
]

YEAR_RANGE = range(2010, 9999)


def main():
    init_log()
    device = auto_set_device()
    
    # [BEGIN] 抽取论文id和详细信息
    if not os.path.isfile('./MAG/output/field_name_2_papers.pkl'):
        field_name_2_id = {field_name: query_field_id_by_name(field_name) for field_name in FIELD_NAMES}

        field_name_2_papers: dict[str, list[dict]] = dict()
        
        for field_name, field_id in tqdm(field_name_2_id.items()):
            other_field_ids = set(field_name_2_id.values())
            other_field_ids.remove(field_id)

            paper_ids = query_cs_paper_by_field(field_id)

            filtered_paper_entries = list()  

            for paper_id in tqdm(paper_ids):
                paper_entry = query_paper_by_id(paper_id)['_source']
                year = int(paper_entry['year'])
                
                if year not in YEAR_RANGE:
                    continue 
                
                field_ids = query_paper_field(paper_id)
                
                if field_ids & other_field_ids:
                    continue 
                
                filtered_paper_entries.append(paper_entry)
                
            field_name_2_papers[field_name] = filtered_paper_entries 
            
        with open('./MAG/output/field_name_2_papers.pkl', 'wb') as fp:
            pickle.dump(field_name_2_papers, fp)
    else:
        with open('./MAG/output/field_name_2_papers.pkl', 'rb') as fp:
            field_name_2_papers = pickle.load(fp)
    # [END]
    
    print("按领域统计论文数量：")
    for field_name in field_name_2_papers:
        print(f"{field_name}: {len(field_name_2_papers[field_name])}")
    print() 
        
    # [BEGIN] 抽取论文作者和引用
    if not os.path.isfile('./MAG/output/field_name_2_papers_with_meta.pkl'):
        for field_name in tqdm(field_name_2_papers):
            for paper_entry in tqdm(field_name_2_papers[field_name]):
                paper_id = int(paper_entry['paper_id'])
                author_ids = query_paper_author(paper_id, max_cnt=3)
                ref_ids = query_paper_reference(paper_id)
                
                paper_entry['author_ids'] = author_ids
                paper_entry['reference_ids'] = ref_ids

        with open('./MAG/output/field_name_2_papers_with_meta.pkl', 'wb') as fp:
            pickle.dump(field_name_2_papers, fp)
    else:
        with open('./MAG/output/field_name_2_papers_with_meta.pkl', 'rb') as fp:
            field_name_2_papers = pickle.load(fp)     
    # [END]
    
    # [BEGIN] 构建异构图
    paper_id_map: dict[int, int] = dict()
    author_id_map: dict[int, int] = dict()
    venue_id_map: dict[int, int] = dict()

    for field_name in field_name_2_papers:
        for paper_entry in field_name_2_papers[field_name]:
            paper_id = int(paper_entry['paper_id'])
            author_ids = paper_entry['author_ids']
            venue_id = paper_entry['conference_series_id'] or paper_entry['journal_id']
            
            if paper_id not in paper_id_map:
                paper_id_map[paper_id] = len(paper_id_map)
                
            if venue_id and venue_id not in venue_id_map:
                venue_id_map[venue_id] = len(venue_id_map)
                
            for author_id in author_ids:
                if author_id not in author_id_map:
                    author_id_map[author_id] = len(author_id_map)
                    
    pp_edge_index = []
    pa_edge_index = []
    pv_edge_index = []
    
    num_papers = len(paper_id_map)
    label_list = [-1] * num_papers 
    year_list = [-1] * num_papers 
    
    for field_nid, field_name in enumerate(field_name_2_papers):
        for paper_entry in field_name_2_papers[field_name]:
            paper_id = int(paper_entry['paper_id'])
            author_ids = paper_entry['author_ids']
            venue_id = paper_entry['conference_series_id'] or paper_entry['journal_id']
            ref_ids = paper_entry['reference_ids']

            label_list[paper_id_map[paper_id]] = field_nid
            year_list[paper_id_map[paper_id]] = int(paper_entry['year'])
            
            for ref_id in ref_ids:
                if ref_id in paper_id_map:
                    pp_edge_index.append((paper_id_map[paper_id], paper_id_map[ref_id]))
                    pp_edge_index.append((paper_id_map[ref_id], paper_id_map[paper_id]))
                    
            for author_id in author_ids:
                pa_edge_index.append((paper_id_map[paper_id], author_id_map[author_id]))
                
            if venue_id:
                pv_edge_index.append((paper_id_map[paper_id], venue_id_map[venue_id]))

    pp_edge_index = tuple(torch.tensor(pp_edge_index, dtype=torch.int64).T)
    pa_edge_index = tuple(torch.tensor(pa_edge_index, dtype=torch.int64).T)
    pv_edge_index = tuple(torch.tensor(pv_edge_index, dtype=torch.int64).T)

    hg = dgl.heterograph({
        ('paper', 'pp', 'paper'): pp_edge_index,
        ('paper', 'pa', 'author'): pa_edge_index,
        ('author', 'ap', 'paper'): (pa_edge_index[1], pa_edge_index[0]),
        ('paper', 'pv', 'venue'): pv_edge_index,
        ('venue', 'vp', 'paper'): (pv_edge_index[1], pv_edge_index[0]),
    })
    
    hg.nodes['paper'].data['label'] = torch.tensor(label_list, dtype=torch.int64)
    hg.nodes['paper'].data['year'] = torch.tensor(year_list, dtype=torch.int64)
    
    print(hg)
    # [END]
    
    # [BEGIN] 统计类间的边数量
    num_outer_edges = 0 
    
    for field_name in field_name_2_papers:
        inner_paper_ids = set() 
        
        for paper_entry in field_name_2_papers[field_name]:
            paper_id = int(paper_entry['paper_id'])
            inner_paper_ids.add(paper_id_map[paper_id])
            
        for paper_entry in field_name_2_papers[field_name]:
            ref_ids = paper_entry['reference_ids']
            
            for ref_id in ref_ids:
                if ref_id in paper_id_map and paper_id_map[ref_id] not in inner_paper_ids:
                    num_outer_edges += 2 
                    
    print(f"类间的边数量：{num_outer_edges}")
    # [END]
    
    # [BEGIN] 生成论文结点特征
    paper_emb = np.zeros((num_papers, 768), dtype=np.float32)
    
    for field_name in tqdm(field_name_2_papers):
        text_list = [paper_entry['original_title'] for paper_entry in field_name_2_papers[field_name]]
        emb = sbert_embedding(text_list)
        assert len(emb) == len(field_name_2_papers[field_name])
        
        for i, paper_entry in enumerate(field_name_2_papers[field_name]):
            nid = paper_id_map[int(paper_entry['paper_id'])] 
            paper_emb[nid] = emb[i]
            
    paper_emb = perform_PCA(feat=paper_emb, out_dim=64)
    
    hg.nodes['paper'].data['feat'] = torch.from_numpy(paper_emb)
    
    with open('./MAG/output/hg.pkl', 'wb') as fp:
        pickle.dump(hg, fp)
    # [END]
    
    
if __name__ == '__main__':
    main() 
