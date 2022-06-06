from util import * 
import os.path as osp
import torch
from torch_geometric.datasets import AMiner

from .model import * 

set_device('cuda:3')

path = '/home/gh/Dataset/AMiner'
dataset = AMiner(path)
data = dataset[0]
data = to_device(data)

metapath = [
    ('author', 'writes', 'paper'),
    ('paper', 'published_in', 'venue'),
    ('venue', 'publishes', 'paper'),
    ('paper', 'written_by', 'author'),
]

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MetaPath2Vec(
    edge_index_dict = data.edge_index_dict, 
    embedding_dim = 128,
    metapath = metapath, 
    walk_length = 50, 
    context_size = 7,
    walks_per_node = 5, 
    num_negative_samples = 5,
    sparse = True,
)
model = to_device(model)

loader = model.get_dataloader(batch_size=128, shuffle=True, num_workers=0)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
# optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01)


def train(epoch, log_steps=100, eval_steps=2000):
    model.train()

    total_loss = 0
    for i, (pos_rw, neg_rw) in enumerate(loader):
        pos_rw = to_device(pos_rw)
        neg_rw = to_device(neg_rw)
        
        optimizer.zero_grad()
        loss = model.calc_loss(pos_rw, neg_rw)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % log_steps == 0:
            print(f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, Loss: {total_loss / log_steps:.4f}')
            total_loss = 0

        # if (i + 1) % 1 == 0:
            f1_micro, f1_macro = test()
            print(f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, f1_micro: {f1_micro:.4f}, f1_macro: {f1_macro:.4f}')


@torch.no_grad()
def test(train_ratio=0.1):
    model.eval()

    emb = model('author')[data['author'].y_index].detach().cpu().numpy() 
    y_true = data['author'].y.cpu().numpy() 

    perm = np.random.permutation(len(emb))
    num_train = int(len(emb) * train_ratio)
    train_idxs = perm[:num_train]
    eval_idxs = perm[num_train:]
    
    train_mask = np.zeros([len(emb)], dtype=bool)
    eval_mask = np.zeros([len(emb)], dtype=bool)
    train_mask[train_idxs] = True 
    eval_mask[eval_idxs] = True 

    return perform_classification(
        emb = emb, 
        label = y_true, 
        train_mask = train_mask, 
        eval_mask = eval_mask, 
    )


for epoch in range(1, 6):
    train(epoch)
    acc = test()
    print(f'Epoch: {epoch}, Accuracy: {acc:.4f}')
