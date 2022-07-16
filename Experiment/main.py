from model import * 

from dl import * 


@dataclass
class Config:
    lr: float = 0.005
    weight_decay: float = 0.0005
    num_epochs: int = 200
    add_self_loop: bool = True 
    val_metapath: Optional[tuple] = None #('pv', 'vp')


def main(config: Config):
    set_cwd(__file__)
    init_log()
    device = auto_set_device()

    hg = load_dgl_graph('../DBLP/output/dblp_2014_2020.pt')
    hg = hg.to(device)
    
    g = hg['pp']
    
    print("原始同构图：")
    print(g)
    print()
    
    feat = g.ndata.pop('feat')
    feat_dim = feat.shape[-1]
    num_nodes = len(feat)
    label_th = g.ndata.pop('label') 
    num_classes = int(torch.max(label_th)) + 1 
    year_th = g.ndata.pop('year')

    print("按年份统计论文数量：")
    year2cnt = defaultdict(int)
    for year in year_th:
        year2cnt[int(year)] += 1 
    pprint(year2cnt) 
    print()
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[year_th <= 2015] = True 
    val_mask = ~train_mask
    
    # [BEGIN] 重新连接测试集结点
    if config.val_metapath:
        mp_g = dgl.metapath_reachable_graph(g=hg, metapath=config.val_metapath)
        
        train_nids = set(train_mask.nonzero().squeeze().tolist())
        val_nids = set(val_mask.nonzero().squeeze().tolist())
        
        g_adj_list = to_adj_list(g)
        mp_g_adj_list = to_adj_list(mp_g)
        
        for src_nid, dest_nids in g_adj_list.items():
            if src_nid in val_nids:
                dest_nids = set(dest_nids)
                dest_nids -= val_nids
                mp_dest_nids = set(mp_g_adj_list[src_nid])
                dest_nids.update(mp_dest_nids - train_nids)
                
                g_adj_list[src_nid] = list(dest_nids)
                
        g_src_index, g_dest_index = to_torch_sparse(g_adj_list).indices()

        corrupt_g = dgl.graph((g_src_index, g_dest_index), num_nodes=num_nodes).to(device)
        
        print("重新连接后的同构图：")
        print(corrupt_g)
        print() 
    else:
        corrupt_g = g 
    # [END]
    
    if config.add_self_loop:
        corrupt_g = dgl.remove_self_loop(corrupt_g)
        corrupt_g = dgl.add_self_loop(corrupt_g)
    
    # model = GAT(
    #     GAT.Param(
    #         in_dim = feat_dim,
    #         out_dim = num_classes,
    #     )
    # )
    model = GraphSAGE(
        in_dim = feat_dim,
        out_dim = num_classes,
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    wandb.init(project='Experiment', config=asdict(config))
    
    wandb.config['raw_g'] = g 
    wandb.config['corrupt_g'] = corrupt_g

    best_val_f1_micro = 0. 

    for epoch in tqdm(range(1, config.num_epochs + 1)):
        # Train        
        loss = model.train_graph(g=corrupt_g, feat=feat, mask=train_mask, label=label_th)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Val & Test
        val_f1_micro, val_f1_macro = model.eval_graph(g=corrupt_g, feat=feat, mask=val_mask, label=label_th)

        if val_f1_micro > best_val_f1_micro:
            best_val_f1_micro = val_f1_micro 
            wandb.summary['best_val_f1_micro'] = best_val_f1_micro
        
        log_multi(
            wandb_log = True, 
            epoch = epoch, 
            loss = float(loss), 
            val_f1_micro = val_f1_micro,
            val_f1_macro = val_f1_macro,
        )
    
    
if __name__ == '__main__':
    main(
        Config()
    ) 
