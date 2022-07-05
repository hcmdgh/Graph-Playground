from .conn import * 
from pprint import pprint 


def query_paper_by_id(paper_id: int) -> dict:
    resp = requests.get(
        url = f"{ES_HOST}/mag_paper/_doc/{paper_id}",
        auth = ES_AUTH,
    ) 
    
    if resp.status_code not in range(200, 300):
        return dict() 
    
    resp_json = resp.json() 
    
    return resp_json


def query_paper_by_field(
    field_id: int,
    max_cnt: int = 10000,
) -> set[int]:
    resp = requests.get(
        url = f"{ES_HOST}/mag_paper_field/_search",
        auth = ES_AUTH,
        json = {
            'query': {
                'match': {
                    'field_id': field_id, 
                },
            },
            'size': max_cnt, 
            'sort': [{ 'citation_count': { 'order': 'desc' } }],
        },
    )
    
    resp_json = resp.json() 
    
    if not resp.status_code in range(200, 300):
        pprint(resp_json)
        raise RuntimeError

    entries = resp_json['hits']['hits']
    
    paper_ids = set(entry['_source']['paper_id'] for entry in entries)
    
    return paper_ids 


def query_paper_reference(paper_id: int) -> set[int]:
    resp = requests.get(
        url = f"{ES_HOST}/mag_paper_reference/_search",
        auth = ES_AUTH,
        json = {
            'query': {
                'match': {
                    'paper_id': paper_id, 
                },
            },
            'size': 999,
        },
    )
    
    assert resp.status_code in range(200, 300)
    
    resp_json = resp.json() 

    entries = resp_json['hits']['hits']
    
    reference_ids = set(int(entry['_source']['paper_reference_id']) for entry in entries)
        
    return reference_ids
