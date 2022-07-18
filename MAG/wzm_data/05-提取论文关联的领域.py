from mag_util import * 
import pickle 
from typing import Any 
from collections import defaultdict 


def main():
    field_map: dict[int, dict[str, Any]] = dict() 
    
    for field_entry in scan_file('field'):
        field_id = field_entry['id']
        field_map[field_id] = field_entry

    with open('./wzm_data/pkl/filtered_paper_id_set_2.pkl', 'rb') as fp:
        paper_id_set = pickle.load(fp)
        
    paper_field_map: dict[int, set[str]] = defaultdict(set)

    for paper_field_entry in scan_file('paper_field'):
        paper_id = paper_field_entry['paper_id']
        field_id = paper_field_entry['field_id']

        if paper_id in paper_id_set:
            field_name = field_map[field_id]['display_name']
            field_level = field_map[field_id]['level']
            
            if field_level <= 2:
                paper_field_map[paper_id].add(field_name)
                
    with open('./wzm_data/pkl/paper_field_map.pkl', 'wb') as fp:
        pickle.dump(paper_field_map, fp)
    
    
if __name__ == '__main__':
    main() 
