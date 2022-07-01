from .conn import * 
from typing import Optional, Iterator 
from tqdm import tqdm 
from collections import defaultdict
import pickle 


def read_tsv(file_path: str,
             num_rows: Optional[int] = None) -> Iterator[list[str]]:
    with open(file_path, 'r', encoding='utf-8') as fp:
        for line in tqdm(fp, total=num_rows, desc=file_path):
            columns = [x.strip() for x in line.split('\t')]
            yield columns


def extract_paper_by_conference(conference_names: set[str]) -> dict[str, dict[int, list]]:
    conn, cursor = get_conn_cursor()
    
    cid_2_cname = dict()
    
    for conference_name in conference_names:
        cursor.execute(
            "SELECT * FROM mag.conference_series WHERE normalized_name = %s",
            [conference_name],
        )
        
        conference_id = cursor.fetchone()['id']
        
        cid_2_cname[conference_id] = conference_name
        
    print(cid_2_cname)
    
    paper_detail_dict: dict[str, dict[int, list]] = defaultdict(dict)
    
    for cols in read_tsv('/home/Dataset/MAG/mag_20211108/mag/Papers.txt', num_rows=2_6945_1039):
        if not cols[0] or not cols[12]:
            continue
        
        paper_id = int(cols[0])
        conference_id = int(cols[12])
        
        if conference_id in cid_2_cname:
            conference_name = cid_2_cname[conference_id]
            
            paper_detail_dict[conference_name][paper_id] = cols 
            
            # print(f"{conference_name}: {len(paper_detail_dict[conference_name])}")

    with open('./MAG/output/paper_detail_by_conference.pkl', 'wb') as fp:
        pickle.dump(paper_detail_dict, fp)
            

def main():
    extract_paper_by_conference({
        'AAAI',
        'KDD',
        'CVPR',
        'ACL',
        'NeurIPS',
        'VR',
    })