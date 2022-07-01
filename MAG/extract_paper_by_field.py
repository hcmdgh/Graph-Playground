from typing import Optional, Iterator 
from tqdm import tqdm 
import pymysql 
import random 
import pickle 
from collections import defaultdict
import os 


def read_tsv(file_path: str,
             num_rows: Optional[int] = None) -> Iterator[list[str]]:
    with open(file_path, 'r', encoding='utf-8') as fp:
        for line in tqdm(fp, total=num_rows, desc=file_path):
            columns = [x.strip() for x in line.split('\t')]
            yield columns


def extract_paper_by_field(field_ids: set[int]):
    if not os.path.isfile('./MAG/output/fid_2_pids.pkl'):
        fid_2_pids: dict[int, set[int]] = defaultdict(set)
        pid_set: set[int] = set() 
        
        for cols in read_tsv('/home/Dataset/MAG/mag_20211108/advanced/PaperFieldsOfStudy.txt', num_rows=15_4656_1902):
            paper_id = int(cols[0])
            field_id = int(cols[1])

            if field_id in field_ids:
                fid_2_pids[field_id].add(paper_id)
                pid_set.add(paper_id)
                
        with open('./MAG/output/fid_2_pids.pkl', 'wb') as fp:
            pickle.dump(fid_2_pids, fp)
    else:
        with open('./MAG/output/fid_2_pids.pkl', 'rb') as fp:
            fid_2_pids = pickle.load(fp)
            
        pid_set = set() 
        
        for pids in fid_2_pids.values():
            pid_set.update(pids)
        
    pid_2_paper: dict[int, list] = dict() 
        
    for cols in read_tsv('/home/Dataset/MAG/mag_20211108/mag/Papers.txt', num_rows=2_6945_1039):
        paper_id = int(cols[0])
        
        if paper_id in pid_set:
            pid_2_paper[paper_id] = cols 
    
    with open('./MAG/output/pid_2_paper.pkl', 'wb') as fp:
        pickle.dump(pid_2_paper, fp)  


def main():
    
    extract_paper_by_field({2780922921, 2780276568, 83209312, 2777648619})
