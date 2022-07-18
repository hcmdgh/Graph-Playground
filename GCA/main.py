from model import * 
from util import * 
from core import * 

from dl import * 


def main(config: Config):
    set_cwd(__file__)
    init_log()
    device = auto_set_device()
    
    core = GCA_Core(config)
    
    train_mask, val_mask, test_mask = split_train_val_test_set(
        total_cnt = core.num_nodes,
        train_ratio = config.train_val_test_ratio[0],
        val_ratio = config.train_val_test_ratio[1],
        test_ratio = config.train_val_test_ratio[2], 
    )

    optimizer = torch.optim.Adam(core.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    wandb.init(project='GCA', config=asdict(config))

    metric_recorder = MetricRecorder()

    for epoch in range(1, config.num_epochs + 1):
        loss = core.train_epoch()
        
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step() 

        if epoch % 100:
            metric_recorder.record(
                epoch = epoch,
                log = True,
                wandb_log = True,
                loss = loss, 
            )
        else:
            eval_res = core.eval_epoch(
                label = core.label,
                train_mask = train_mask,
                val_mask = val_mask,
                test_mask = test_mask,
            )

            metric_recorder.record(
                epoch = epoch,
                log = True,
                wandb_log = True,
                loss = loss, 
                val_f1_micro = eval_res['val_f1_micro'], 
                val_f1_macro = eval_res['val_f1_macro'],
                test_f1_micro = eval_res['test_f1_micro'], 
                test_f1_macro = eval_res['test_f1_macro'], 
            )

    metric_recorder.best_record(
        'val_f1_micro',
        log = True,
        wandb_log = True,         
    )


if __name__ == '__main__':
    main(
        Config(
            graph = load_pyg_dataset('amazon-computers'),
        )
    )
