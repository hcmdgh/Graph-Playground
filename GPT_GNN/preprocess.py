from util import * 
from gensim.models import Word2Vec
import transformers

DATASET_ROOT = '/home/Dataset/GPT-GNN/OAG/MAG_0919_CS'

BERT_BATCH_SIZE = 64 


def read_tsv(path: str,
             num_rows: Optional[int] = None,
             drop_first: bool = True) -> Iterator[list[str]]:
    with open(path, 'r', encoding='utf-8') as fp:
        if drop_first:
            fp.readline()
        
        for line in tqdm(fp, total=num_rows):
            cols = [col.strip() for col in line.split('\t')]

            yield cols 


def main():
    init_log()
    device = auto_set_device()
    
    # [BEGIN] 统计论文被引次数
    if not os.path.isfile('./GPT_GNN/cache/pid_2_num_cited.pkl'):
        pid_2_num_cited: dict[int, int] = defaultdict(int)
        
        for cols in read_tsv(os.path.join(DATASET_ROOT, 'PR_CS_20190919.tsv'), num_rows=3144_1552):
            cited_paper_id = int(cols[1])
            pid_2_num_cited[cited_paper_id] += 1 
            
        with open('./GPT_GNN/cache/pid_2_num_cited.pkl', 'wb') as fp:
            pickle.dump(pid_2_num_cited, fp)
    else:
        with open('./GPT_GNN/cache/pid_2_num_cited.pkl', 'rb') as fp:
            pid_2_num_cited = pickle.load(fp)
    # [END]
    
    # [BEGIN] 筛选论文
    if not os.path.isfile('./GPT_GNN/cache/pid_2_paper.pkl'):
        pid_2_paper: dict[int, dict[str, Any]] = dict() 
        
        for cols in read_tsv(os.path.join(DATASET_ROOT, 'Papers_CS_20190919.tsv'), num_rows=559_7606):
            if not (cols[0] and cols[1] and cols[2] and cols[3] and cols[4]):
                continue
            
            paper_id = int(cols[0])
            year = int(cols[1])
            num_citations = pid_2_num_cited[paper_id]
            citation_bound = min(2020 - year, 20)
            
            if num_citations < citation_bound or year < 1900:
                continue 
            
            paper_entry = { 
                'id': paper_id,
                'title': cols[2],
                'year': year, 
            }
            
            pid_2_paper[paper_id] = paper_entry
            
        with open('./GPT_GNN/cache/pid_2_paper.pkl', 'wb') as fp:
            pickle.dump(pid_2_paper, fp)
    else:
        with open('./GPT_GNN/cache/pid_2_paper.pkl', 'rb') as fp:
            pid_2_paper = pickle.load(fp)
    # [END]
    
    # [BEGIN] 论文生成向量表征（标题）
    if not os.path.isfile('./GPT_GNN/cache/pid_2_paper_with_emb.pkl'):
        bert_tokenizer = transformers.XLNetTokenizer.from_pretrained('xlnet-base-cased')
        bert_model = transformers.XLNetModel.from_pretrained(
            'xlnet-base-cased',
            output_hidden_states = True,
            output_attentions = True,
        ).to(device)
        
        def bert_embedding(text_batch: list[str]) -> FloatArray:
            with torch.no_grad():
                inputs = bert_tokenizer(
                    text_batch, 
                    padding = True,
                    truncation = True,
                    max_length = 512,
                    return_tensors = 'pt',
                ).to(device)
                
                outputs = bert_model(**inputs)

                last_hidden_states = outputs.last_hidden_state

                out = torch.mean(last_hidden_states, dim=1)

                return out.detach().cpu().numpy() 

        paper_ids = list(pid_2_paper.keys())
        
        for i in tqdm(range(0, len(paper_ids), BERT_BATCH_SIZE)):
            text_batch = [pid_2_paper[paper_id]['title'] for paper_id in paper_ids[i: i + BERT_BATCH_SIZE]]

            emb_batch = bert_embedding(text_batch)
            
            for j, paper_id in enumerate(paper_ids[i: i + BERT_BATCH_SIZE]):
                pid_2_paper[paper_id]['emb'] = emb_batch[j] 

        # for cols in read_tsv(os.path.join(DATASET_ROOT, 'PAb_CS_20190919.tsv'), num_rows=454_2602):
        #     paper_id = int(cols[0])
            
        #     if paper_id in pid_2_paper:
        #         input_ids = torch.tensor([bert_tokenizer.encode(pid_2_paper[paper_id]['title'])]).to(device)[:, :64]

        #         if len(input_ids[0]) < 4:
        #             continue

        #         all_hidden_states, all_attentions = bert_model(input_ids)[-2:]

        #         emb = (all_hidden_states[-2][0] * all_attentions[-2][0].mean(dim=0).mean(dim=0).view(-1, 1)).sum(dim=0)
        #         pid_2_paper[paper_id]['emb'] = emb.detach().cpu().numpy() 
                
        with open('./GPT_GNN/cache/pid_2_paper.pkl_with_emb', 'wb') as fp:
            pickle.dump(pid_2_paper, fp)
    else:
        with open('./GPT_GNN/cache/pid_2_paper.pkl_with_emb', 'rb') as fp:
            pid_2_paper = pickle.load(fp)
    # [END]

if __name__ == '__main__':
    main() 
