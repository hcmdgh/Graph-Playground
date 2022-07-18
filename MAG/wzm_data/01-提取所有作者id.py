from tqdm import tqdm 
import pickle 

RELATION_PATH = '/storage/wzm/relations.txt'


def main():
    author_id_set = set() 
    paper_id_set = set() 
    
    with open(RELATION_PATH, 'r', encoding='utf-8') as fp:
        for line in tqdm(fp, total=5_2481_7518):
            line = line.strip() 
            
            if not line:
                continue 
            
            cols = line.split('\t')

            paper_id = int(cols[0])
            author_id = int(cols[1])
            
            paper_id_set.add(paper_id)
            author_id_set.add(author_id)
    
    print(f"论文数量：{len(paper_id_set)}")
    print(f"作者数量：{len(author_id_set)}")
    
    with open('./wzm_data/output/paper_id_set.pkl', 'wb') as fp:
        pickle.dump(paper_id_set, fp)
        
    with open('./wzm_data/output/author_id_set.pkl', 'wb') as fp:
        pickle.dump(author_id_set, fp)    
            
            
if __name__ == '__main__':
    main() 
