from query import *  
from prettytable import PrettyTable

# fp = open('./output.txt', 'w', encoding='utf-8')
fp = None 


def draw_table(entries: list[dict]):
    table = PrettyTable(field_names=['field_id', 'field_name', 'num_papers'])
    
    entries.sort(key=lambda x: -x['num_papers'])
    
    for entry in entries:
        table.add_row([entry['field_id'], entry['field_name'], entry['num_papers']])
        
    print(table, file=fp)


def main(field_name: str):
    field_id = query_field_id_by_name(field_name) 
    sub_field_ids = query_sub_field(field_id)
    super_field_ids = query_super_field(field_id)
    
    print("父级领域：", file=fp)
    
    entries = []
    
    for field_id in super_field_ids:
        field_detail = query_field_by_id(field_id)
        entries.append({'field_id': field_id, 'field_name': field_detail['display_name'], 'num_papers': field_detail['paper_count']})
        
    draw_table(entries)
    
    
    print("同级领域：", file=fp)
    
    if len(super_field_ids) == 1:
        field_id = super_field_ids.pop() 
        field_ids = query_sub_field(field_id)
        
        entries = []
    
        for field_id in field_ids:
            field_detail = query_field_by_id(field_id)
            entries.append({'field_id': field_id, 'field_name': field_detail['display_name'], 'num_papers': field_detail['paper_count']})
            
        draw_table(entries)
    
    
    print("子级领域：", file=fp)
    
    entries = []
    
    for field_id in sub_field_ids:
        field_detail = query_field_by_id(field_id)
        entries.append({'field_id': field_id, 'field_name': field_detail['display_name'], 'num_papers': field_detail['paper_count']})
        
    draw_table(entries)
    
    
if __name__ == '__main__':
    main('Natural language processing')
