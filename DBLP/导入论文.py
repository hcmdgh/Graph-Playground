from conn import * 

from typing import Optional 
import json 
from tqdm import tqdm 
from pprint import pprint 

BATCH_SIZE = 10000


def json_dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def bulk_insert(*,
                index: str, 
                id_list: Optional[list] = None,
                entry_list: list[dict]):
    if not entry_list:
        return 
    
    if id_list:
        assert len(id_list) == len(entry_list)
                
    request_body = ''
        
    for i, entry in enumerate(entry_list):
        if id_list:
            request_body += json_dump({ 'index': { '_index': index, '_id': id_list[i] } }) + '\n'
        else:
            request_body += json_dump({ 'index': { '_index': index } }) + '\n'

        request_body += json_dump(entry) + '\n'

    resp = requests.post(
        url = f"{ES_HOST}/_bulk",
        auth = ES_AUTH, 
        headers = {"Content-Type": "application/json;charset=UTF-8"},
        data = request_body.encode('utf-8'),
    )    
    
    assert resp.status_code in range(200, 300)


def main():
    resp = requests.delete(
        url = f"{ES_HOST}/dblp_v11_paper",
        auth = ES_AUTH, 
    ) 
    assert resp.status_code in range(200, 300) or resp.status_code == 404 
    
    resp = requests.put(
        url = f"{ES_HOST}/dblp_v11_paper",
        auth = ES_AUTH, 
        json = {
            "mappings": {
                "properties": {
                    "id": { "type": "keyword" },
                    "authors": {
                        "properties": {
                            "id": { "type": "keyword" },
                            "name": { "type": "keyword" },
                            "org": { "type": "keyword" },
                        }
                    },
                    "doc_type": { "type": "keyword" },
                    "doi": { "type": "keyword" },
                    "fos": {
                        "properties": {
                            "name": { "type": "keyword" },
                            "w": { "type": "float" },
                        }
                    },
                    "issue": { "type": "keyword" },
                    "n_citation": { "type": "long" },
                    "page_end": { "type": "keyword" },
                    "page_start": { "type": "keyword" },
                    "publisher": { "type": "keyword" },
                    "references": { "type": "keyword" },
                    "title": { "type": "keyword", "ignore_above": 256 },
                    "abstract": { "type": "text" },
                    "venue": {
                        "properties": {
                            "raw": { "type": "keyword" },
                            "id": { "type": "keyword" },
                        }
                    },
                    "volume": { "type": "keyword" },
                    "year": { "type": "long" },
                }
            }
        }
    ) 
    assert resp.status_code in range(200, 300)
    
    with open('/home/Dataset/CitationNetworkDataset/dblp-v11/dblp_papers_v11.txt', 'r', encoding='utf-8') as fp:
        doc_batch = [] 
        id_batch = [] 
        
        for line in tqdm(fp):
            doc = json.loads(line)
            
            if 'indexed_abstract' in doc:
                indexed_abstract = doc.pop('indexed_abstract')
                
                word_len = indexed_abstract['IndexLength']
                word_list = [''] * word_len 
                
                for word, idxs in indexed_abstract['InvertedIndex'].items():
                    for idx in idxs:
                        word_list[idx] = word 
                        
                abstract = ' '.join(word_list)
            else:
                abstract = None 
                
            doc['abstract'] = abstract
            
            doc_batch.append(doc)
            id_batch.append(doc['id'])
            
            if len(doc_batch) >= BATCH_SIZE:
                bulk_insert(index='dblp_v11_paper', id_list=id_batch, entry_list=doc_batch) 
                
                doc_batch.clear()
                id_batch.clear() 

        bulk_insert(index='dblp_v11_paper', id_list=id_batch, entry_list=doc_batch) 
        

if __name__ == '__main__':
    main() 
