from util import * 

YEAR_RANGE_1 = range(0, 2015)
YEAR_RANGE_2 = range(2015, 2019)


def main():
    with open('./MAG/output/g.pkl', 'rb') as fp:
        g = pickle.load(fp)
        
    year_np = g.ndata['year'].numpy() 
        
    year_2_cnt = defaultdict(int)
    
    for year in year_np:
        year_2_cnt[year] += 1 
        
    pprint(year_2_cnt)


    nid_list_1 = []
    nid_list_2 = []
    nid_list_3 = []
    
    for nid, year in enumerate(year_np):
        if year in YEAR_RANGE_1:
            nid_list_1.append(nid)
        elif year in YEAR_RANGE_2:
            nid_list_2.append(nid)
        else:
            nid_list_3.append(nid)
    
    g_1 = dgl.node_subgraph(g, nid_list_1)
    g_2 = dgl.node_subgraph(g, nid_list_2)
    g_3 = dgl.node_subgraph(g, nid_list_3)

    print(g)
    print(g_1)
    print(g_2)
    print(g_3)
    
    with open('./MAG/output/g_1.pkl', 'wb') as fp:
        pickle.dump(g_1, fp)
        
    with open('./MAG/output/g_2.pkl', 'wb') as fp:
        pickle.dump(g_2, fp)

    with open('./MAG/output/g_3.pkl', 'wb') as fp:
        pickle.dump(g_3, fp)
