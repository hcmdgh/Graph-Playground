import pickle 
from tqdm import tqdm 


def main():
    with open('./wzm_data/pkl/filtered_author_paper_map.pkl', 'rb') as fp:
        author_paper_map = pickle.load(fp)
        
    print(f"筛选前作者数量：{len(author_paper_map)}")

    for author_id, author_entry in tqdm(list(author_paper_map.items())):
        if len(author_entry['paper_id_set']) < 30:
            del author_paper_map[author_id]
            
    paper_id_set = set() 
    
    for author_id, author_entry in tqdm(author_paper_map.items()):
        paper_id_set.update(author_entry['paper_id_set'])
    
    print(f"筛选后作者数量：{len(author_paper_map)}")
    print(f"筛选后论文数量：{len(paper_id_set)}")
    
    with open('./wzm_data/pkl/filtered_author_paper_map_2.pkl', 'wb') as fp:
        pickle.dump(author_paper_map, fp)
        
    with open('./wzm_data/pkl/filtered_paper_id_set_2.pkl', 'wb') as fp:
        pickle.dump(paper_id_set, fp)


if __name__ == '__main__':
    main() 
