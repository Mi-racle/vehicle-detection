import os

import mysql.connector


def clear_dir(dir_name: str):
    if os.path.exists(dir_name):
        for name in os.listdir(dir_name):
            if os.path.isdir(f'{dir_name}/{name}'):
                clear_dir(f'{dir_name}/{name}')
            else:
                os.remove(f'{dir_name}/{name}')


def truncate_table(host: str, port: int, user: str, password: str, database: str, table: str):
    conn = mysql.connector.connect(
        **{
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
    )
    cursor = conn.cursor(dictionary=True)

    query = f'''TRUNCATE {table}'''

    cursor.execute(query)
    conn.commit()

    if conn and conn.is_connected():
        conn.close()


if __name__ == '__main__':
    clear_dir('D:/xxs-signs/vehicle-detection/src/runs')
    truncate_table(
        host='localhost',
        port=3306,
        user='root',
        password='123456',
        database='db_banma_traffic_analysis',
        table='tbl_result'
    )
