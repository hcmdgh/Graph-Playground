from .graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from .graphmae.datasets.data_util import load_dataset
from .graphmae.evaluation import node_classification_evaluation
from .graphmae.models.edcoder import PreModel

from util import * 

__all__ = ['GraphMAE_pipeline']


def pretrain(model, graph, feat, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)

    epoch_iter = tqdm(range(max_epoch))

    for epoch in epoch_iter:
        model.train()

        loss, loss_dict = model(graph, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

        if (epoch + 1) % 200 == 0:
            node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, mute=True)

    # return best_model
    return model


def GraphMAE_pipeline(
    graph: dgl.DGLGraph,
    seeds: list[int] = [1428],
    lr: float = 0.001, 
    lr_f: float = 0.01,
    num_hidden: int = 512,
    num_heads: int = 4,
    num_out_heads: int = 1,
    weight_decay: float = 2e-4,
    weight_decay_f: float = 1e-4,
    max_epoch: int = 1500,
    max_epoch_f: int = 300,
    mask_rate: float = 0.5,
    num_layers: int = 2,
    negative_slope: float = 0.2,
    encoder_type: str = 'gat',
    decoder_type: str = 'gat',
    activation: str = 'prelu',
    in_drop: float = 0.2,
    attn_drop: float = 0.1,
    linear_prob: bool = True,
    residual: bool = False,
    concat_hidden: bool = False,
    norm: Optional[str] = None,
    loss_fn: str = 'sce',
    drop_edge_rate: float = 0.0,
    optimizer_type: str = 'adam',
    replace_rate: float = 0.05,
    alpha_l: int = 3,
    use_scheduler: bool = True,
):
    init_log()
    device = auto_set_device()
    
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    
    feat = graph.ndata['feat']
    feat_dim = feat.shape[-1]
    label = graph.ndata['label']
    num_classes = int(torch.max(label)) + 1 

    acc_list = []
    estp_acc_list = []
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        logger = None

        model = PreModel(
            in_dim = feat_dim,
            num_hidden = num_hidden,
            num_layers = num_layers,
            nhead = num_heads,
            nhead_out = num_out_heads,
            activation = activation,
            feat_drop = in_drop,
            attn_drop = attn_drop,
            negative_slope = negative_slope,
            residual = residual,
            encoder_type = encoder_type,
            decoder_type = decoder_type,
            mask_rate = mask_rate,
            norm = norm,
            loss_fn = loss_fn,
            drop_edge_rate = drop_edge_rate,
            replace_rate = replace_rate,
            alpha_l = alpha_l,
            concat_hidden = concat_hidden,
        ).to(device)

        optimizer = create_optimizer(optimizer_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                    # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None
            
        x = graph.ndata["feat"]
        # if not load_model:
        model = pretrain(model, graph, x, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
        model = model.cpu()

        # if load_model:
        #     logging.info("Loading Model ... ")
        #     model.load_state_dict(torch.load("checkpoint.pt"))
        # if save_model:
        #     logging.info("Saveing Model ...")
        #     torch.save(model.state_dict(), "checkpoint.pt")
        
        model = model.to(device)
        model.eval()

        final_acc, estp_acc = node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob)
        acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)

        if logger is not None:
            logger.finish()

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")
