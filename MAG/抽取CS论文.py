import pickle 
from tqdm import tqdm 
import os 
import json 

CS_FIELD_ID = 41008148

OTHER_FIELD_ID = {
    15744967,
    17744445,
    33923547,
    39432304,
    71924100,
    86803240,
    95457728,
    121332964,
    127313418,
    127413603,
    138885662,
    142362112,
    144024400,
    144133560,
    162324750,
    185592680,
    192562407,
    205649164,
}


def main():
    # with open('./MAG/pkl/paper_field_map.pkl', 'rb') as fp:
    #     paper_field_map = pickle.load(fp)
        
    if not os.path.isfile('./MAG/pkl/cs_paper_ids.pkl'):
        cs_paper_ids = [] 
            
        for paper_id, field_ids in tqdm(paper_field_map.items()):
            field_ids = set(field_ids)
            
            if CS_FIELD_ID in field_ids and not (field_ids & OTHER_FIELD_ID):
                cs_paper_ids.append(paper_id)
                
        with open('./MAG/pkl/cs_paper_ids.pkl', 'wb') as fp:
            pickle.dump(cs_paper_ids, fp)
    else:
        with open('./MAG/pkl/cs_paper_ids.pkl', 'rb') as fp:
            cs_paper_ids = pickle.load(fp)

    print(f"仅含CS领域的论文数量：{len(cs_paper_ids)}")
    exit()
    
    cs_paper_field_map: dict[int, list[int]] = dict() 
    
    for paper_id in tqdm(cs_paper_ids):
        assert paper_id in paper_field_map
        cs_paper_field_map[paper_id] = paper_field_map[paper_id]
        
    with open('./MAG/pkl/cs_paper_field_map.pkl', 'wb') as fp:
        pickle.dump(cs_paper_field_map, fp)
    
    
if __name__ == '__main__':
    main() 
