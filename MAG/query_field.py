from .conn import * 
from tqdm import tqdm 


def query_field_by_id(field_id: int) -> dict:
    conn, cursor = get_conn_cursor()
    
    cursor.execute(
        "SELECT * FROM mag.field_of_study WHERE id = %s",
        [field_id],
    )
    
    res_list = list(cursor.fetchall()) 
    
    assert len(res_list) == 1 
    
    return dict(res_list[0])


def query_field_by_name(field_name: str) -> dict:
    conn, cursor = get_conn_cursor()

    field_name = field_name.lower().strip() 
    
    cursor.execute(
        "SELECT * FROM mag.field_of_study WHERE normalized_name = %s",
        [field_name],
    )
    
    res_list = list(cursor.fetchall()) 
    
    assert len(res_list) == 1 
    
    return dict(res_list[0])


def query_child_field(field_name: str) -> list[dict]:
    conn, cursor = get_conn_cursor()
    
    field_id = query_field_by_name(field_name)['id']
    
    cursor.execute(
        "SELECT * FROM mag.field_of_study_children WHERE field_of_study_id = %s",
        [field_id], 
    )
    
    child_field_list = []
    
    for entry in cursor.fetchall():
        child_field_id = entry['child_field_of_study_id']

        child_field_list.append(query_field_by_id(child_field_id))
    
    return child_field_list 


def query_field_paper_id(field_ids: set[int],
                         from_file: bool = False) -> set[int]:
    if not from_file:
        conn, cursor = get_conn_cursor()
        
        cursor.execute(
            "SELECT * FROM mag.paper_field_of_study WHERE field_of_study_id = %s",
            [field_id],
        )
        
        paper_id_set = set()
        
        for entry in cursor.fetchall():
            paper_id_set.add(entry['paper_id'])
            
        return paper_id_set 
    else:
        with open('/home/Dataset/MAG/mag_20211108/advanced/PaperFieldsOfStudy.txt', 'r') as fp:
            paper_id_set = set()
            
            for line in tqdm(fp, total=15_4656_1902):
                cols = [col.strip() for col in line.split('\t')] 
                
                field_id = int(cols[1])
                
                if field_id in field_ids:
                    paper_id = int(cols[0])
                    
                    paper_id_set.add(paper_id)
                    
            return paper_id_set 
    