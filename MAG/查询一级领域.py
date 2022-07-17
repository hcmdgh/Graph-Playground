import requests 
from pprint import pprint 
from prettytable import PrettyTable

ES_HOST = 'http://192.168.1.153:19200'
ES_AUTH = ('elastic', 'HghFq3QLPb7Qv5')


def main():
    resp = requests.get(
        url = f"{ES_HOST}/mag_field/_search",
        auth = ES_AUTH,
        json = {
            'query': {
                'term': {
                    'level': 0, 
                }
            },
            'size': 9999, 
        },
    )
    
    if resp.status_code not in range(200, 300):
        pprint(resp.json())
        raise RuntimeError
    
    resp_json = resp.json() 
    
    field_list = [] 
    
    for entry in resp_json['hits']['hits']:
        field_list.append({
            'name': entry['_source']['display_name'],
            'paper_count': entry['_source']['paper_count'],
        })
        
    field_list.sort(key=lambda x: x['paper_count'], reverse=True)
    
    table = PrettyTable(['name', 'paper_count'])
    
    for entry in field_list:
        table.add_row([entry['name'], entry['paper_count']])
        
    print(table)


if __name__ == '__main__':
    main() 
