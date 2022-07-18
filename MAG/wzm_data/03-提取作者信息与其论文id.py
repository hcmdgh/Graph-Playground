import pickle 
from tqdm import tqdm 
from typing import Any 
from collections import defaultdict


def main():
    with open('./wzm_data/pkl/filtered_author_id_set.pkl', 'rb') as fp:
        filtered_author_id_set = pickle.load(fp)
        
    author_dict: dict[int, dict[str, Any]] = dict() 
        
    with open('/storage/wzm/relations.txt', 'r', encoding='utf-8') as fp:
        for line in tqdm(fp, total=5_2481_7518):
            cols = line.strip().split('\t')

            try:
                paper_id = int(cols[0])
                author_id = int(cols[1])
                author_name = cols[4].strip() 
                author_org = cols[5].strip() 
            except (IndexError, ValueError):
                pass 
            else:
                if author_id in filtered_author_id_set:
                    if author_id not in author_dict:
                        author_dict[author_id] = {
                            'name': None,
                            'org': None,
                            'paper_id_set': set(), 
                        }
                        
                    author_entry = author_dict[author_id]
                    
                    if author_name and author_org:
                        author_entry['name'] = author_name
                        author_entry['org'] = author_org
                        
                    author_entry['paper_id_set'].add(paper_id)
                    
    paper_id_set = set() 
    
    for author_id, author_entry in tqdm(list(author_dict.items())):
        if not author_entry['name'] or not author_entry['org']:
            del author_dict[author_id]
        else:
            paper_id_set.update(author_entry['paper_id_set'])
    
    print(f"筛选后的作者数量：{len(author_dict)}")
    print(f"筛选后的论文数量：{len(paper_id_set)}")
    
    with open('./wzm_data/pkl/filtered_author_paper_map.pkl', 'wb') as fp:
        pickle.dump(author_dict, fp)
        
    with open('./wzm_data/pkl/filtered_paper_id_set.pkl', 'wb') as fp:
        pickle.dump(paper_id_set, fp)
        

if __name__ == '__main__':
    main() 
