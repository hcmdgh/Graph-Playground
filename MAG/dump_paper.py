from .conn import * 
from typing import Optional, Iterator 
from tqdm import tqdm 
import pymysql 
import random 
import pickle 


def read_tsv(file_path: str) -> Iterator[list[str]]:
    with open(file_path, 'r', encoding='utf-8') as fp:
        for line in fp:
            columns = [x.strip() for x in line.split('\t')]
            yield columns


def dump_table(table_name: str,
               file_path: str,
               id_set: set[int], 
               total_cnt: Optional[int] = None):
    conn, cursor = get_conn_cursor()

    for row_i, columns in enumerate(tqdm(read_tsv(file_path), desc=table_name, total=total_cnt)):
        n_col = len(columns)

        columns = [None if not x else x for x in columns]
        
        id = int(columns[0])
        
        if id not in id_set:
            continue 

        insert_sql = f"INSERT INTO {table_name} VALUES ({','.join(['%s'] * n_col)})"

        try:
            cursor.execute(insert_sql, columns)
        except pymysql.err.IntegrityError:
            pass

    conn.commit()


def main():
    with open('./MAG/output/paper_id_set_ds.pkl', 'rb') as fp:
        paper_id_set = pickle.load(fp)
    
    dump_table(
        table_name = 'mag.paper',
        file_path = '/home/Dataset/MAG/mag_20211108/mag/Papers.txt',
        id_set = paper_id_set,
        total_cnt = 2_6945_1039,
    )


if __name__ == '__main__':
    main() 
