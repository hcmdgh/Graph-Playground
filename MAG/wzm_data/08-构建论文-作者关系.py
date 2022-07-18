import pickle 
import random 
from tqdm import tqdm 
from collections import defaultdict


def main():
    with open('./wzm_data/pkl/filtered_author_paper_map_2.pkl', 'rb') as fp:
        author_paper_map = pickle.load(fp)
        
    paper_author_map: dict[int, set[tuple[str, str]]] = defaultdict(set)
        
    for author_id, entry in tqdm(author_paper_map.items()):
        author_name = entry['name']
        author_org = entry['org']

        for paper_id in entry['paper_id_set']:
            paper_author_map[paper_id].add((author_name, author_org))
            
    with open('./wzm_data/pkl/paper_author_map.pkl', 'wb') as fp:
        pickle.dump(paper_author_map, fp)


if __name__ == '__main__':
    main() 
