from conn import * 


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
