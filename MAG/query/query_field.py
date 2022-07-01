from .conn import * 


def query_field_by_id(field_id: int) -> dict:
    resp = requests.get(
        url = f"{ES_HOST}/mag_field/_doc/{field_id}",
        auth = ES_AUTH,
    ) 
    
    if resp.status_code not in range(200, 300):
        return dict() 
    
    resp_json = resp.json() 
    
    return resp_json['_source']


def query_field_by_name(field_name: str) -> tuple[list[int], list[dict]]:
    field_name = field_name.lower().strip() 
    
    resp = requests.get(
        url = f"{ES_HOST}/mag_field/_search",
        auth = ES_AUTH,
        json = {
            'query': {
                'match': {
                    'normalized_name': field_name, 
                },
            },
        },
    )
    
    assert resp.status_code in range(200, 300)
    
    resp_json = resp.json() 

    entries = resp_json['hits']['hits']
    
    field_ids = [int(entry['_id']) for entry in entries]
    
    return field_ids, entries 


def query_field_id_by_name(field_name: str) -> int:
    field_ids, _ = query_field_by_name(field_name)
    
    if len(field_ids) == 1:
        return field_ids.pop() 
    elif not field_ids:
        raise FileNotFoundError 
    else:
        raise AssertionError


def query_sub_field(field_id: int) -> set[int]:
    resp = requests.get(
        url = f"{ES_HOST}/mag_field_children/_search",
        auth = ES_AUTH,
        json = {
            'query': {
                'match': {
                    'field_id': field_id, 
                },
            },
            'size': 999,
        },
    )
    
    assert resp.status_code in range(200, 300)
    
    resp_json = resp.json() 

    entries = resp_json['hits']['hits']
    
    sub_field_ids = set(int(entry['_source']['child_field_id']) for entry in entries)
        
    return sub_field_ids


def query_super_field(field_id: int) -> set[int]:
    resp = requests.get(
        url = f"{ES_HOST}/mag_field_children/_search",
        auth = ES_AUTH,
        json = {
            'query': {
                'match': {
                    'child_field_id': field_id, 
                },
            },
            'size': 999,
        },
    )
    
    assert resp.status_code in range(200, 300)
    
    resp_json = resp.json() 

    entries = resp_json['hits']['hits']
    
    super_field_ids = set(int(entry['_source']['field_id']) for entry in entries)
        
    return super_field_ids


def query_paper_field(paper_id: int) -> set[int]:
    resp = requests.get(
        url = f"{ES_HOST}/mag_paper_field/_search",
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
    
    field_ids = set(int(entry['_source']['field_id']) for entry in entries)
        
    return field_ids
