from util import * 


def visualize(labels, g):
    pos = nx.spring_layout(g, seed=1)
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    nx.draw_networkx(
        g, 
        pos = pos, 
        node_size = 50, 
        cmap = plt.get_cmap('coolwarm'),
        node_color = labels, 
        edge_color = 'k',
        arrows = False, 
        width = 0.5, 
        style = 'dotted', 
        with_labels = False,
    )
    
    plt.savefig('./test.png', dpi=200)