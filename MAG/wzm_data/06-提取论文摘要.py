from mag_util import * 
import pickle 
from typing import Any 
from tqdm import tqdm 
import json 


def main():
    with open('./wzm_data/pkl/filtered_paper_id_set_2.pkl', 'rb') as fp:
        paper_id_set = pickle.load(fp)

    paper_abstract_map: dict[int, str] = dict()
    
    with open('/storage/wzm/publish_abstract.txt', 'r', encoding='utf-8') as fp:
        for line in tqdm(fp, total=6235_5709):
            entry = json.loads(line)
            
            paper_id = int(entry['paperId'])
            abstract = entry['abs']
            
            if paper_id in paper_id_set:
                paper_abstract_map[paper_id] = abstract

    with open('./wzm_data/pkl/paper_abstract_map.pkl', 'wb') as fp:
        pickle.dump(paper_abstract_map, fp)
    

if __name__ == '__main__':
    main() 
