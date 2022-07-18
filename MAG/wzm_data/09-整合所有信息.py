from dl import *  


def main():
    set_cwd(__file__)
    
    with open('./pkl/paper_detail_map.pkl', 'rb') as fp:
        paper_detail_map = pickle.load(fp) 
    print("加载完成：1")
        
    with open('./pkl/paper_abstract_map.pkl', 'rb') as fp:
        paper_abstract_map = pickle.load(fp)
    print("加载完成：2")

    with open('./pkl/paper_author_map.pkl', 'rb') as fp:
        paper_author_map = pickle.load(fp)
    print("加载完成：3")

    with open('./pkl/paper_field_map.pkl', 'rb') as fp:
        paper_field_map = pickle.load(fp)
    print("加载完成：4")

    paper_id_set = set(paper_author_map.keys())

    with open('./output/paper_rich.pkl', 'wb') as fp_out:
        for paper_id in tqdm(paper_id_set):
            paper_entry = {
                'detail': paper_detail_map.pop(paper_id, None),
                'abstract': paper_abstract_map.pop(paper_id, None),
                'author': paper_author_map.pop(paper_id, None),
                'field': paper_field_map.pop(paper_id, None),
            } 
            
            pickle.dump(paper_entry, fp_out)
            
    print(f"完成，总共导入论文数量：{len(paper_id_set)}")


if __name__ == '__main__':
    main() 
