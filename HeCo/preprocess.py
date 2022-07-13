from dl import * 


def preprocess(dataset_name: str):
    if dataset_name == 'ACM':
        paper_feat = sp.load_npz('/home/gh/Graph-Playground/HeCo/data/acm/p_feat.npz').toarray().astype(np.float32)
        np.save('/home/Dataset/HeCo/ACM/feat_paper.npy', paper_feat)

        label = np.load('/home/gh/Graph-Playground/HeCo/data/acm/labels.npy').astype(np.int64)
        np.save('/home/Dataset/HeCo/ACM/label_paper.npy', label)

        pa_adj_raw = np.load('/home/gh/Graph-Playground/HeCo/data/acm/nei_a.npy', allow_pickle=True)
        pa_adj = to_coo_mat({src_nid: dest_nids.tolist() for src_nid, dest_nids in enumerate(pa_adj_raw)}) 
        sp.save_npz('/home/Dataset/HeCo/ACM/adj_paper_author.npz', pa_adj)

        ps_adj_raw = np.load('/home/gh/Graph-Playground/HeCo/data/acm/nei_s.npy', allow_pickle=True)
        ps_adj = to_coo_mat({src_nid: dest_nids.tolist() for src_nid, dest_nids in enumerate(ps_adj_raw)}) 
        sp.save_npz('/home/Dataset/HeCo/ACM/adj_paper_subject.npz', ps_adj)

        pap_adj = to_coo_mat(sp.load_npz('/home/gh/Graph-Playground/HeCo/data/acm/pap.npz'))
        psp_adj = to_coo_mat(sp.load_npz('/home/gh/Graph-Playground/HeCo/data/acm/psp.npz'))
        sp.save_npz('/home/Dataset/HeCo/ACM/adj_paper_author_paper.npz', pap_adj)
        sp.save_npz('/home/Dataset/HeCo/ACM/adj_paper_subject_paper.npz', psp_adj)

        pos = to_coo_mat(sp.load_npz('/home/gh/Graph-Playground/HeCo/data/acm/pos.npz'))
        sp.save_npz('/home/Dataset/HeCo/ACM/positive_sample_paper.npz', pos)

        train_idx_20 = np.load('/home/gh/Graph-Playground/HeCo/data/acm/train_20.npy')
        val_idx_20 = np.load('/home/gh/Graph-Playground/HeCo/data/acm/val_20.npy')
        test_idx_20 = np.load('/home/gh/Graph-Playground/HeCo/data/acm/test_20.npy')
        train_idx_40 = np.load('/home/gh/Graph-Playground/HeCo/data/acm/train_40.npy')
        val_idx_40 = np.load('/home/gh/Graph-Playground/HeCo/data/acm/val_40.npy')
        test_idx_40 = np.load('/home/gh/Graph-Playground/HeCo/data/acm/test_40.npy')
        train_idx_60 = np.load('/home/gh/Graph-Playground/HeCo/data/acm/train_60.npy')
        val_idx_60 = np.load('/home/gh/Graph-Playground/HeCo/data/acm/val_60.npy')
        test_idx_60 = np.load('/home/gh/Graph-Playground/HeCo/data/acm/test_60.npy')
        
        paper_num_nodes = len(paper_feat)
        train_mask_20 = np.zeros(paper_num_nodes, dtype=bool)
        val_mask_20 = np.zeros(paper_num_nodes, dtype=bool)
        test_mask_20 = np.zeros(paper_num_nodes, dtype=bool)
        train_mask_40 = np.zeros(paper_num_nodes, dtype=bool)
        val_mask_40 = np.zeros(paper_num_nodes, dtype=bool)
        test_mask_40 = np.zeros(paper_num_nodes, dtype=bool)
        train_mask_60 = np.zeros(paper_num_nodes, dtype=bool)
        val_mask_60 = np.zeros(paper_num_nodes, dtype=bool)
        test_mask_60 = np.zeros(paper_num_nodes, dtype=bool)
        
        train_mask_20[train_idx_20] = True 
        val_mask_20[val_idx_20] = True 
        test_mask_20[test_idx_20] = True 
        train_mask_40[train_idx_40] = True 
        val_mask_40[val_idx_40] = True 
        test_mask_40[test_idx_40] = True 
        train_mask_60[train_idx_60] = True 
        val_mask_60[val_idx_60] = True 
        test_mask_60[test_idx_60] = True 
        
        np.save('/home/Dataset/HeCo/ACM/train_mask_20.npy', train_mask_20)
        np.save('/home/Dataset/HeCo/ACM/val_mask_20.npy', val_mask_20)
        np.save('/home/Dataset/HeCo/ACM/test_mask_20.npy', test_mask_20)
        np.save('/home/Dataset/HeCo/ACM/train_mask_40.npy', train_mask_40)
        np.save('/home/Dataset/HeCo/ACM/val_mask_40.npy', val_mask_40)
        np.save('/home/Dataset/HeCo/ACM/test_mask_40.npy', test_mask_40)
        np.save('/home/Dataset/HeCo/ACM/train_mask_60.npy', train_mask_60)
        np.save('/home/Dataset/HeCo/ACM/val_mask_60.npy', val_mask_60)
        np.save('/home/Dataset/HeCo/ACM/test_mask_60.npy', test_mask_60)

        # [BEGIN] 数据集划分统计
        print(train_idx_20.shape, np.min(train_idx_20), np.max(train_idx_20))
        print(val_idx_20.shape, np.min(val_idx_20), np.max(val_idx_20))
        print(test_idx_20.shape, np.min(test_idx_20), np.max(test_idx_20))
        print(train_idx_40.shape, np.min(train_idx_40), np.max(train_idx_40))
        print(val_idx_40.shape, np.min(val_idx_40), np.max(val_idx_40))
        print(test_idx_40.shape, np.min(test_idx_40), np.max(test_idx_40))
        print(train_idx_60.shape, np.min(train_idx_60), np.max(train_idx_60))
        print(val_idx_60.shape, np.min(val_idx_60), np.max(val_idx_60))
        print(test_idx_60.shape, np.min(test_idx_60), np.max(test_idx_60))
        # [END]
        
        paper_label = np.load('/home/Dataset/HeCo/ACM/label_paper.npy')
        paper_feat = np.load('/home/Dataset/HeCo/ACM/feat_paper.npy')
        
        paper_num_nodes = 4019
        author_num_nodes = 7167
        subject_num_nodes = 60 
        assert paper_num_nodes == len(paper_feat)

        pa_adj = sp.load_npz('/home/Dataset/HeCo/ACM/adj_paper_author.npz')
        ps_adj = sp.load_npz('/home/Dataset/HeCo/ACM/adj_paper_subject.npz')

        paper_pos = sp.load_npz('/home/Dataset/HeCo/ACM/positive_sample_paper.npz')
        
        train_mask_20 = np.load('/home/Dataset/HeCo/ACM/train_mask_20.npy')
        val_mask_20 = np.load('/home/Dataset/HeCo/ACM/val_mask_20.npy')
        test_mask_20 = np.load('/home/Dataset/HeCo/ACM/test_mask_20.npy')
        train_mask_40 = np.load('/home/Dataset/HeCo/ACM/train_mask_40.npy')
        val_mask_40 = np.load('/home/Dataset/HeCo/ACM/val_mask_40.npy')
        test_mask_40 = np.load('/home/Dataset/HeCo/ACM/test_mask_40.npy')
        train_mask_60 = np.load('/home/Dataset/HeCo/ACM/train_mask_60.npy')
        val_mask_60 = np.load('/home/Dataset/HeCo/ACM/val_mask_60.npy')
        test_mask_60 = np.load('/home/Dataset/HeCo/ACM/test_mask_60.npy')

        hg = dgl.heterograph(
            data_dict = {
                ('paper', 'pa', 'author'): (pa_adj.row, pa_adj.col),
                ('author', 'ap', 'paper'): (pa_adj.col, pa_adj.row),
                ('paper', 'ps', 'subject'): (ps_adj.row, ps_adj.col),
                ('subject', 'sp', 'paper'): (ps_adj.col, ps_adj.row),
            }
        )
        assert hg.num_nodes('paper') == paper_num_nodes
        assert hg.num_nodes('author') == author_num_nodes
        assert hg.num_nodes('subject') == subject_num_nodes
        
        hg.nodes['paper'].data['label'] = torch.from_numpy(paper_label).to(torch.int64)
        hg.nodes['paper'].data['feat'] = torch.from_numpy(paper_feat).to(torch.float32)
        hg.nodes['paper'].data['train_mask_20'] = torch.from_numpy(train_mask_20).to(torch.bool)
        hg.nodes['paper'].data['val_mask_20'] = torch.from_numpy(val_mask_20).to(torch.bool)
        hg.nodes['paper'].data['test_mask_20'] = torch.from_numpy(test_mask_20).to(torch.bool)
        hg.nodes['paper'].data['train_mask_40'] = torch.from_numpy(train_mask_40).to(torch.bool)
        hg.nodes['paper'].data['val_mask_40'] = torch.from_numpy(val_mask_40).to(torch.bool)
        hg.nodes['paper'].data['test_mask_40'] = torch.from_numpy(test_mask_40).to(torch.bool)
        hg.nodes['paper'].data['train_mask_60'] = torch.from_numpy(train_mask_60).to(torch.bool)
        hg.nodes['paper'].data['val_mask_60'] = torch.from_numpy(val_mask_60).to(torch.bool)
        hg.nodes['paper'].data['test_mask_60'] = torch.from_numpy(test_mask_60).to(torch.bool)

        save_dgl_graph(hg, '/home/Dataset/GengHao/HeCo/ACM.pt')
    else:
        raise AssertionError 


if __name__ == '__main__':
    preprocess('ACM')
