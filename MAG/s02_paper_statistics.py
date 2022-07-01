import pickle 
from pprint import pprint 
from collections import defaultdict


def main():
    with open('./MAG/output/paper_detail_by_conference.pkl', 'rb') as fp:
        paper_detail_dict = pickle.load(fp)
        
    # 筛选某一年之后的论文
    for cname in paper_detail_dict:
        paper_detail_dict[cname] = {
            paper_id: paper_entry 
            for paper_id, paper_entry in paper_detail_dict[cname].items()
            if int(paper_entry[7]) >= 2010 
        }
    
    with open('./MAG/output/paper_detail_by_conference_filtered.pkl', 'wb') as fp:
        pickle.dump(paper_detail_dict, fp)
    
    cname_2_cnt = dict() 
    
    for cname in paper_detail_dict:
        cname_2_cnt[cname] = len(paper_detail_dict[cname])
        
    pprint(cname_2_cnt)
    
    print(f"total: {sum(cname_2_cnt.values())}")
    
    cname_2_year_2_cnt = dict() 
    
    for cname in paper_detail_dict:
        cname_2_year_2_cnt[cname] = defaultdict(int)
        
        for paper_entry in paper_detail_dict[cname].values():
            year = int(paper_entry[7])
            cname_2_year_2_cnt[cname][year] += 1 
        
    pprint(cname_2_year_2_cnt)
