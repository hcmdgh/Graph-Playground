from mag_util import * 

import pickle 


def main():
    with open('./wzm_data/pkl/author_id_set.pkl', 'rb') as fp:
        author_id_set = pickle.load(fp) 
        
    filtered_author_id_set = set() 

    for author_entry in scan_file('author'):
        author_id = author_entry['id']
        
        if author_id in author_id_set:
            paper_count = author_entry['paper_count']
            
            if paper_count and paper_count >= 30:
                filtered_author_id_set.add(author_id)
                
    print(f"筛选后的作者数量：{len(filtered_author_id_set)}")     
        
    with open('./wzm_data/pkl/filtered_author_id_set.pkl', 'wb') as fp:
        pickle.dump(filtered_author_id_set, fp)
        
        
if __name__ == '__main__':
    main() 
