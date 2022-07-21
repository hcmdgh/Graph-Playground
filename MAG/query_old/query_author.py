from .conn import * 
from typing import Optional


def query_paper_author(paper_id: int,
                       max_cnt: int = 999) -> set[int]:
    resp = requests.get(
        url = f"{ES_HOST}/mag_paper_author_affiliation/_search",
        auth = ES_AUTH,
        json = {
            'query': {
                'bool': {
                    'must': [
                        {
                            'match': {
                                'paper_id': paper_id, 
                            }
                        },
                        {
                            'range': {
                                'author_sequence_number': {
                                    'lte': max_cnt, 
                                }
                            }    
                        },
                    ]
                }
            },
            'size': 999,
        },
    )
    
    assert resp.status_code in range(200, 300)
    
    resp_json = resp.json() 

    entries = resp_json['hits']['hits']
    
    author_ids = set(int(entry['_source']['author_id']) for entry in entries)
        
    return author_ids
