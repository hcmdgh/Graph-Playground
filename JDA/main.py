from util import *
from .model import * 

DATASET_ROOT = '/home/Dataset/WangJindong/Office-Caltech10'

DOMAINS = ['caltech_surf_10.mat', 'amazon_surf_10.mat', 'webcam_surf_10.mat', 'dslr_surf_10.mat']


def main():
    init_log()
    
    for domain_S in DOMAINS:
        for domain_T in DOMAINS:
            if domain_S != domain_T:
                mat_S = sio.loadmat(os.path.join(DATASET_ROOT, domain_S))
                mat_T = sio.loadmat(os.path.join(DATASET_ROOT, domain_T))

                feat_S = mat_S['feas']
                feat_T = mat_T['feas']
                label_S = mat_S['label'] - 1
                label_T = mat_T['label'] - 1 
                label_S = label_S.reshape(-1)
                label_T = label_T.reshape(-1)
                
                feat_all = np.concatenate([feat_S, feat_T], axis=0)
                label_all = np.concatenate([label_S, label_T], axis=0)
                
                mask_S = np.zeros(len(feat_S) + len(feat_T), dtype=bool)
                mask_T = np.zeros(len(feat_S) + len(feat_T), dtype=bool)
                mask_S[:len(feat_S)] = True 
                mask_T[len(feat_S):] = True 
                
                
                val_f1_micro, val_f1_macro, _, _ = KNeighbors_multiclass_classification(
                    feat = feat_all,
                    label = label_all,
                    train_mask = mask_S,
                    val_mask = mask_T,
                )
                
                logging.info(f"No TCA, KNeighbors: {domain_S} -> {domain_T}: val_f1_micro = {val_f1_micro:.4f}, val_f1_macro = {val_f1_macro:.4f}")
                
                val_f1_micro, val_f1_macro, _, _ = xgb_multiclass_classification(
                    feat = feat_all,
                    label = label_all,
                    train_mask = mask_S,
                    val_mask = mask_T,
                )
                
                logging.info(f"No TCA, XGBoost: {domain_S} -> {domain_T}: val_f1_micro = {val_f1_micro:.4f}, val_f1_macro = {val_f1_macro:.4f}")

                
                tca = JDA(kernel_type='linear')

                out_S, out_T = tca.fit(feat_S=feat_S, label_S=label_S, feat_T=feat_T)

                out_all = np.concatenate([out_S, out_T], axis=0)
                
                val_f1_micro, val_f1_macro, _, _ = KNeighbors_multiclass_classification(
                    feat = out_all,
                    label = label_all,
                    train_mask = mask_S,
                    val_mask = mask_T,
                )
                
                logging.info(f"TCA, KNeighbors: {domain_S} -> {domain_T}: val_f1_micro = {val_f1_micro:.4f}, val_f1_macro = {val_f1_macro:.4f}")
                
                val_f1_micro, val_f1_macro, _, _ = xgb_multiclass_classification(
                    feat = out_all,
                    label = label_all,
                    train_mask = mask_S,
                    val_mask = mask_T,
                )
                
                logging.info(f"TCA, XGBoost: {domain_S} -> {domain_T}: val_f1_micro = {val_f1_micro:.4f}, val_f1_macro = {val_f1_macro:.4f}")
                

if __name__ == '__main__':
    main() 
