import logging

import mysql.connector
from mysql.connector import Error

from db.db_config import DB_CONFIG


def get_online_task_by_id(task_id: int):
    connection = None
    cursor = None

    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)

            table = 'tbl_task_run'
            query = f'''SELECT * FROM {table} WHERE id = %s'''

            cursor.execute(query, (task_id,))
            logging.info(f'Entry successfully selected from {table}')

            return cursor.fetchone()

    except Error as e:
        connection = None
        logging.info(f'Error: {e}')

    finally:
        if connection and connection.is_connected():
            if cursor:
                cursor.close()
            connection.close()
            logging.info('MySQL connection closed')
