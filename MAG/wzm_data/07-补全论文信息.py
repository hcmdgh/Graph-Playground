from mag_util import * 
import pickle 
from typing import Any 
from tqdm import tqdm 
import json 


def main():
    with open('./wzm_data/pkl/filtered_paper_id_set_2.pkl', 'rb') as fp:
        paper_id_set = pickle.load(fp)
    
    paper_detail_map: dict[int, dict[str, Any]] = dict() 
        
    for paper_entry in scan_file('paper'):
        paper_id = paper_entry['id']
        
        if paper_id in paper_id_set:
            paper_detail_map[paper_id] = paper_entry 
            
    with open('./wzm_data/pkl/paper_detail_map.pkl', 'wb') as fp:
        pickle.dump(paper_detail_map, fp)


if __name__ == '__main__':
    main() 
