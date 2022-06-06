from util import * 

_EPS = 1e-15


class MetaPath2Vec(nn.Module):
    def __init__(self,
                 edge_index_dict: dict[tuple[str, str, str], tuple[IntTensor, IntTensor]],
                 embedding_dim: int,
                 metapath: list[tuple[str, str, str]],
                 walk_length: int,
                 context_size: int,
                 walks_per_node: int = 1,
                 num_negative_samples: int = 1,
                 num_nodes_dict: Optional[dict[str, int]] = None,
                 sparse: bool = False):
        super().__init__()
        
        if num_nodes_dict is None:
            num_nodes_dict = dict() 
            
            for etype, edge_index in edge_index_dict.items():
                src_type = etype[0]
                N = int(torch.max(edge_index[0])) + 1 
                num_nodes_dict[src_type] = max(num_nodes_dict.get(src_type, N), N)
                
                dest_type = etype[-1]
                N = int(torch.max(edge_index[1])) + 1 
                num_nodes_dict[dest_type] = max(num_nodes_dict.get(dest_type, N), N)
                
        assert walk_length + 1 >= context_size
        assert not (walk_length > len(metapath) and metapath[0][0] != metapath[-1][-1]) 
        
        self.embedding_dim = embedding_dim
        self.metapath = metapath
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples
        self.num_nodes_dict = num_nodes_dict
        
        # [BEGIN] 异构图每种类型的结点重新编码
        ntypes = set(x[0] for x in metapath) | set(x[-1] for x in metapath)
        ntypes = sorted(ntypes)
        
        idx = 0
        self.nid_bound_dict: dict[str, tuple[int, int]] = dict()

        for ntype in ntypes:
            begin = idx 
            idx += num_nodes_dict[ntype]
            end = idx 
            
            self.nid_bound_dict[ntype] = (begin, end)
        # [END]
        
        # self.edge_index_dict = {
        #     etype: EdgeIndex(
        #         edge_index = edge_index,
        #         first_nid_S = self.idx_bound_dict[etype[0]][0],
        #         num_nodes_S = self.idx_bound_dict[etype[0]][1] - self.idx_bound_dict[etype[0]][0],
        #         first_nid_T = self.idx_bound_dict[etype[-1]][0],
        #         num_nodes_T = self.idx_bound_dict[etype[-1]][1] - self.idx_bound_dict[etype[-1]][0],
        #     )
        #     for etype, edge_index in edge_index_dict.items() 
        # } 
        
        # 重新编码后的edge_index
        self.edge_index_dict = {
            etype: EdgeIndex((edge_index[0] + self.nid_bound_dict[etype[0]][0], 
                              edge_index[1] + self.nid_bound_dict[etype[-1]][0]))
            for etype, edge_index in edge_index_dict.items()
        }
        
        self.num_nodes = idx 
        
        self.embedding = nn.Embedding(
            num_embeddings = self.num_nodes, 
            embedding_dim = embedding_dim, 
            sparse = sparse, 
        )
        
        self.reset_parameters() 
        
    def reset_parameters(self):
        self.embedding.reset_parameters() 
        
    def forward(self,
                ntype: str) -> FloatTensor:
        begin, end = self.nid_bound_dict[ntype]
        emb = self.embedding.weight[begin:end]
        
        return emb 
    
    def _positive_sample(self,
                         nid_batch: IntTensor) -> IntTensor:
        # nid_batch: int[batch_size]
                 
        # origin_nids: int[(batch_size * walks_per_node)] 
        origin_nids = nid_batch.repeat(self.walks_per_node)

        paths = [origin_nids]
        
        for i in range(self.walk_length):
            etype = self.metapath[i % len(self.metapath)]
            
            edge_index = self.edge_index_dict[etype]

            # -> int[(batch_size * walks_per_node)] 
            neighbor_nids = edge_index.sample_neighbors(
                nid_batch = origin_nids,
                num_neighbors = 1,
            ).view(-1)
            
            paths.append(neighbor_nids)
            
            origin_nids = neighbor_nids 

        # path_th: int[(batch_size * walks_per_node) x (walk_length + 1)]
        path_th = torch.stack(paths, dim=-1)

        walks = []
        num_walks_per_path = 1 + self.walk_length + 1 - self.context_size

        for j in range(num_walks_per_path):
            walks.append(path_th[:, j: j + self.context_size])

        # walk_th: int[(num_walks_per_path * batch_size * walks_per_node) x context_size]
        walk_th = torch.cat(walks, dim=0)

        return walk_th  

    def _negative_sample(self, 
                         nid_batch: IntTensor) -> IntTensor:
        # nid_batch: int[batch_size]
                         
        # nid_batch: int[(batch_size * walks_per_node * num_negative_samples)]
        nid_batch = nid_batch.repeat(self.walks_per_node * self.num_negative_samples)

        paths = [nid_batch]

        for i in range(self.walk_length):
            etype = self.metapath[i % len(self.metapath)]

            sampled_nids = torch.randint(
                low = self.nid_bound_dict[etype[-1]][0], 
                high = self.nid_bound_dict[etype[-1]][1],
                size = [len(nid_batch)],
                dtype = torch.int64,
            )

            paths.append(sampled_nids)

        # path_th: int[(batch_size * walks_per_node * num_negative_samples) x (walk_length + 1)]
        path_th = torch.stack(paths, dim=-1)

        walks = []
        num_walks_per_path = 1 + self.walk_length + 1 - self.context_size

        for j in range(num_walks_per_path):
            walks.append(path_th[:, j: j + self.context_size])

        # walk_th: int[(batch_size * walks_per_node * num_negative_samples * num_walks_per_path) x context_size]
        walk_th = torch.cat(walks, dim=0)

        return walk_th 

    def _sample(self,
                nid_batch: Union[IntTensor, list[int]]) -> tuple[IntTensor, IntTensor]:
        if not isinstance(nid_batch, Tensor):
            nid_batch = torch.tensor(nid_batch, dtype=torch.int64)
            
        return self._positive_sample(nid_batch), self._negative_sample(nid_batch)
    
    def calc_loss(self,
                  pos_walks: IntTensor,
                  neg_walks: IntTensor) -> FloatScalarTensor:
        # [BEGIN] Positive Loss 
        # start: int[pos_batch_size]
        start = pos_walks[:, 0]
        
        # rest: int[pos_batch_size x (context_size - 1)]
        rest = pos_walks[:, 1:]
        
        # start_emb: [pos_batch_size x 1 x embedding_dim]
        start_emb = self.embedding(start).view(-1, 1, self.embedding_dim)

        # rest_emb: [pos_batch_size x (context_size - 1) x embedding_dim]
        rest_emb = self.embedding(rest.reshape(-1)).view(pos_walks.shape[0], -1, self.embedding_dim)

        # similarity: [(pos_batch_size * (context_size - 1))]
        similarity = torch.sum(
            start_emb * rest_emb, 
            dim = -1, 
        ).view(-1)
        
        pos_loss = torch.mean(
            -torch.log(
                torch.sigmoid(similarity) + _EPS 
            )
        )
        # [END]
        
        # [BEGIN] Negative Loss 
        # start: int[neg_batch_size]
        start = neg_walks[:, 0]
        
        # rest: int[neg_batch_size x (context_size - 1)]
        rest = neg_walks[:, 1:]
        
        # start_emb: [neg_batch_size x 1 x embedding_dim]
        start_emb = self.embedding(start).view(-1, 1, self.embedding_dim)

        # rest_emb: [neg_batch_size x (context_size - 1) x embedding_dim]
        rest_emb = self.embedding(rest.reshape(-1)).view(neg_walks.shape[0], -1, self.embedding_dim)

        # similarity: [(neg_batch_size * (context_size - 1))]
        similarity = torch.sum(
            start_emb * rest_emb, 
            dim = -1, 
        ).view(-1)
        
        neg_loss = torch.mean(
            -torch.log(
                1. - torch.sigmoid(similarity) + _EPS 
            )
        )
        # [END]
        
        return pos_loss + neg_loss 

    def get_dataloader(self, **kwargs) -> DataLoader:
        start_ntype = self.metapath[0][0]
        
        return DataLoader(
            dataset = range(*self.nid_bound_dict[start_ntype]), 
            collate_fn = self._sample, 
            **kwargs, 
        )
