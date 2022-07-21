from query import * 

from dl import *  

FIELD_NAMES = [
    'Computer vision', 
    'Computer network', 
    'Data mining', 
    'Natural language processing', 
    'Software engineering',
    'Computer security',
    'Computer hardware',
]

YEAR_RANGE = (2010, 9999)

MAX_CNT_PER_FIELD = 5000  


def main():
    set_cwd(__file__)
    
    paper_list = []
    
    field_name_2_id = {
        field_name: query_field_id_by_name(field_name)
        for field_name in FIELD_NAMES
    }
    
    label_list: list[int] = [] 
    year_list: list[int] = [] 
    
    for field_nid, field_name in enumerate(tqdm(FIELD_NAMES)):
        field_id = field_name_2_id[field_name] 
        other_field_ids = set(field_name_2_id.values()) - {field_id}
        
        _paper_list = query_cs_paper_by_field(
            year_range = YEAR_RANGE,
            field_id = field_id,
            max_cnt = MAX_CNT_PER_FIELD, 
            excluded_field_ids = other_field_ids, 
        )
        
        for paper_entry in _paper_list:
            label_list.append(field_nid)
            year_list.append(paper_entry['year'])

        paper_list.extend(_paper_list)
        
    assert len(paper_list) == len(label_list)
    print(f"筛选出的论文总数：{len(paper_list)}")
    print() 
    
    print("按年份统计论文数量：")
    year2cnt = defaultdict(int)
    for year in year_list:
        year2cnt[year] += 1
    pprint(year2cnt)
    print()
        
    paper_nid_map: dict[int, int] = dict()
    author_nid_map: dict[str, int] = dict()
    org_nid_map: dict[str, int] = dict()
    venue_nid_map: dict[int, int] = dict() 
    
    paper_author_edge_list: list[tuple[int, int]] = []
    paper_org_edge_list: list[tuple[int, int]] = []
    paper_venue_edge_list: list[tuple[int, int]] = []
        
    for paper_nid, paper_entry in enumerate(tqdm(paper_list)):
        paper_id = paper_entry['id']
        paper_nid_map[paper_id] = paper_nid

        venue_id = paper_entry.get('journal_id') or paper_entry.get('conference_series_id')

        if venue_id:
            if venue_id not in venue_nid_map:
                venue_nid_map[venue_id] = len(venue_nid_map)
                
            venue_nid = venue_nid_map[venue_id]
            
            paper_venue_edge_list.append((paper_nid, venue_nid))
        
        for author_org_entry in query_paper_author_org(paper_id):
            author = author_org_entry.get('original_author') 
            org = author_org_entry.get('original_affiliation') 
            
            if author:
                if author not in author_nid_map:
                    author_nid_map[author] = len(author_nid_map)

                author_nid = author_nid_map[author]
                
                paper_author_edge_list.append((paper_nid, author_nid))

            if org:
                if org not in org_nid_map:
                    org_nid_map[org] = len(org_nid_map)

                org_nid = org_nid_map[org]
                
                paper_org_edge_list.append((paper_nid, org_nid))
                
    paper_paper_edge_list: list[tuple[int, int]] = []
                
    for paper_nid, paper_entry in enumerate(tqdm(paper_list)):
        paper_id = paper_entry['id']
        
        for ref_id in query_paper_reference_id(paper_id):
            ref_nid = paper_nid_map.get(ref_id)
            
            if ref_nid:
                paper_paper_edge_list.append((paper_nid, ref_nid))
                paper_paper_edge_list.append((ref_nid, paper_nid))
        
    pa_edge_index = tuple(torch.tensor(paper_author_edge_list, dtype=torch.int64).T)
    po_edge_index = tuple(torch.tensor(paper_org_edge_list, dtype=torch.int64).T)
    pv_edge_index = tuple(torch.tensor(paper_venue_edge_list, dtype=torch.int64).T)
    pp_edge_index = tuple(torch.tensor(paper_paper_edge_list, dtype=torch.int64).T)
        
    hg = dgl.heterograph(
        {
            ('paper', 'pa', 'author'): pa_edge_index,
            ('author', 'ap', 'paper'): pa_edge_index[::-1],
            ('paper', 'po', 'org'): po_edge_index,
            ('org', 'op', 'paper'): po_edge_index[::-1],
            ('paper', 'pv', 'venue'): pv_edge_index,
            ('venue', 'vp', 'paper'): pv_edge_index[::-1],
            ('paper', 'pp', 'paper'): pp_edge_index, 
        }
    )
    
    hg.nodes['paper'].data['label'] = torch.tensor(label_list, dtype=torch.int64)
    hg.nodes['paper'].data['year'] = torch.tensor(year_list, dtype=torch.int64)

    print(hg)
    print() 
        

if __name__ == '__main__':
    main() 
