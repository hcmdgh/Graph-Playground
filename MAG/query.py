import requests 
from typing import Any 

__all__ = [
    'query_cs_paper_by_field',
    'query_field_by_name',
    'query_field_id_by_name',
    'query_paper_author_org',
    'query_paper_reference_id',
]

ES_HOST = 'http://192.168.1.153:19200'
ES_AUTH = ('elastic', 'HghFq3QLPb7Qv5')


def query_paper_reference_id(paper_id: int) -> set[int]:
    resp = requests.get(
        url = f"{ES_HOST}/mag_paper_reference/_search",
        auth = ES_AUTH, 
        json = {
            'query': {
                'term': {
                    'paper_id': paper_id
                }
            },
            'size': 999, 
        }, 
    ) 
    
    assert resp.status_code in range(200, 300)
    
    resp_json = resp.json() 
    
    reference_id_set = set()  
    
    for entry in resp_json['hits']['hits']:
        reference_id = entry['_source']['paper_reference_id']
        reference_id_set.add(reference_id)
        
    return reference_id_set


def query_paper_author_org(paper_id: int) -> list[dict[str, Any]]:
    resp = requests.get(
        url = f"{ES_HOST}/mag_paper_author_affiliation/_search",
        auth = ES_AUTH, 
        json = {
            'query': {
                'term': {
                    'paper_id': paper_id
                }
            },
            'size': 999, 
        }, 
    ) 
    
    assert resp.status_code in range(200, 300)
    
    resp_json = resp.json() 
    
    author_org_list = [] 
    
    for entry in resp_json['hits']['hits']:
        author_org_entry = entry['_source']
        author_org_list.append(author_org_entry)
        
    return author_org_list 


def query_field_id_by_name(field_name: str) -> int:
    field_entry_list = query_field_by_name(field_name)
    assert len(field_entry_list) == 1 
    
    return field_entry_list[0]['id']    


def query_field_by_name(field_name: str) -> list[dict[str, Any]]:
    resp = requests.get(
        url = f"{ES_HOST}/mag_field/_search",
        auth = ES_AUTH, 
        json = {
            'query': {
                'bool': {
                    'should': [
                        {
                            'term': {
                                'normalized_name': field_name
                            }    
                        },
                        {
                            'term': {
                                'display_name': field_name
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
    
    field_entry_list = [] 
    
    for entry in resp_json['hits']['hits']:
        field_entry = entry['_source']
        field_entry_list.append(field_entry)
        
    return field_entry_list 


def query_cs_paper_by_field(field_id: int,
                            year_range: tuple[int, int] = (0, 9999),
                            excluded_field_ids: set[int] = set(), 
                            max_cnt: int = 1000) -> list[dict[str, Any]]:
    resp = requests.get(
        url = f"{ES_HOST}/mag_cs_paper/_search",
        auth = ES_AUTH, 
        json = {
            'query': {
                'bool': {
                    'must': [
                        {
                            'range': {
                                'year': { 'gte': year_range[0], 'lte': year_range[1] }
                            }
                        },
                        {
                            'term': {
                                'field_id_list': field_id
                            }    
                        },
                        {
                            'bool': {
                                'must_not': [
                                    {
                                        'terms': {
                                            'field_id_list': list(excluded_field_ids)
                                        }
                                    },    
                                ]
                            }
                        },
                        {
                            'terms': {
                                'doc_type': ['Conference', 'Journal'],
                            }    
                        },
                    ]
                }
            },
            'sort': {
                'citation_count': { 'order': 'desc' }
            },
            'size': max_cnt, 
        },
    )
    
    assert resp.status_code in range(200, 300)
    
    resp_json = resp.json() 
    
    paper_entry_list = [] 
    
    for entry in resp_json['hits']['hits']:
        paper_entry = entry['_source']
        paper_entry_list.append(paper_entry)
        
    return paper_entry_list 
