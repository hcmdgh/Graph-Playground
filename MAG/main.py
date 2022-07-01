from util import * 

if __name__ == '__main__':
    init_log()
    device = auto_set_device()
    
    with open('/home/gh/MAG-Project/output/papers.json', 'r', encoding='utf-8') as fp:
        paper_dict = json.load(fp)
        
    text_list = [] 
    label_list = [] 
        
    for label, (conference_name, paper_entries) in enumerate(paper_dict.items()):
        for paper_entry in paper_entries:
            label_list.append(label)
            text_list.append(paper_entry['_source']['original_title'])
            
    label_arr = np.array(label_list)
    
    N = len(label_arr)
    num_train = int(N * 0.8)
    
    shuffled_idxs = np.random.permutation(N)
    train_mask = np.zeros(N, dtype=bool)
    train_mask[shuffled_idxs[:num_train]] = True 
    val_mask = ~train_mask 
    
    for model_name in [
        'all-MiniLM-L12-v2',
        'all-MiniLM-L6-v2',
        'all-distilroberta-v1',
        'all-mpnet-base-v2',
        'distiluse-base-multilingual-cased-v1',
        'distiluse-base-multilingual-cased-v2',
        'multi-qa-MiniLM-L6-cos-v1',
        'multi-qa-distilbert-cos-v1',
        'multi-qa-mpnet-base-dot-v1',
        'paraphrase-MiniLM-L3-v2',
        'paraphrase-albert-small-v2',
        'paraphrase-multilingual-MiniLM-L12-v2',
        'paraphrase-multilingual-mpnet-base-v2',
    ]:
        emb_arr = sbert_embedding(
            model_name = model_name,
            text_list = text_list,
            batch_size = 128, 
        )
        
        xgb_res = xgb_multiclass_classification(
            feat = emb_arr,
            label = label_arr,
            train_mask = train_mask,
            val_mask = val_mask, 
        )

        logging.info(model_name)
        logging.info(str(xgb_res))
        
        emb_arr = perform_PCA(feat=emb_arr, out_dim=64)
        
        # emb_arr = sbert_embedding(
        #     model_name = model_name,
        #     text_list = text_list,
        #     batch_size = 128, 
        # )
        
        xgb_res = xgb_multiclass_classification(
            feat = emb_arr,
            label = label_arr,
            train_mask = train_mask,
            val_mask = val_mask, 
        )

        logging.info(f"{model_name} - PCA")
        logging.info(str(xgb_res))
