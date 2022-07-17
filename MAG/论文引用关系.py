import pickle 
from tqdm import tqdm 
from collections import defaultdict


def main():
    paper_ref_map: dict[int, list[int]] = defaultdict(list)
    
    with open('/home/Dataset/MAG/mag_20211108/mag/PaperReferences.txt', 'r', encoding='utf-8') as fp:
        for line in tqdm(fp, total=19_3203_0797):
            paper_id, ref_id = map(int, line.split())
            paper_ref_map[paper_id].append(ref_id)

    with open('./MAG/pkl/paper_ref_map.pkl', 'wb') as fp:
        pickle.dump(paper_ref_map, fp)
            
            
if __name__ == '__main__':
    main() 
