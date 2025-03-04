import json
import logging

import mysql.connector
from mysql.connector import Error

from db.db_config import DB_CONFIG


def get_group_by_group_id(group_id: int):
    connection = None
    cursor = None

    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)

            table = 'tbl_group'
            query = f'''SELECT * FROM {table} WHERE group_id = %s'''

            cursor.execute(query, (group_id,))
            logging.info(f'Entry successfully selected from {table}')

            group = cursor.fetchone()
            group['args'] = json.loads(group['args'])

            return group

    except Error as e:
        connection = None
        logging.info(f'Error: {e}')

    finally:
        if connection and connection.is_connected():
            if cursor:
                cursor.close()
            connection.close()
            logging.info('MySQL connection closed')