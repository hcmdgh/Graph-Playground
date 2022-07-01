import pymysql 
import pymysql.cursors

__all__ = ['get_conn_cursor'] 

_mysql_conn = None 


class MAGMySQLConnection:
    def __init__(self):
        self.conn = pymysql.connect(
            host = '192.168.1.153',
            port = 14285,
            user = 'root',
            password = 'bDZRt5q8jA99eu',
            database = 'mag',
            charset = 'utf8mb4',
            cursorclass = pymysql.cursors.DictCursor,
        )
        
        self.cursor = self.conn.cursor()
        
    def __del__(self):
        self.cursor.__exit__()
        self.conn.__exit__()


def get_conn_cursor():
    global _mysql_conn 
    
    if _mysql_conn is None:
        _mysql_conn = MAGMySQLConnection()
        
    return _mysql_conn.conn, _mysql_conn.cursor 
