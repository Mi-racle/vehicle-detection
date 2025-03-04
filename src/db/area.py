import json
import logging

import mysql.connector
from mysql.connector import Error

from db.db_config import DB_CONFIG


def get_area_by_area_id(area_id: str):
    connection = None
    cursor = None

    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)

            table = 'tbl_area'
            query = f'''SELECT * FROM {table} WHERE area_id = %s'''

            cursor.execute(query, (area_id,))
            logging.info(f'Entry successfully selected from {table}')

            area = cursor.fetchone()
            area['vertices'] = json.loads(area['vertices']) if area['vertices'] else None
            area['side_lengths'] = json.loads(area['side_lengths'])

            return area

    except Error as e:
        connection = None
        logging.info(f'Error: {e}')

    finally:
        if connection and connection.is_connected():
            if cursor:
                cursor.close()
            connection.close()
            logging.info('MySQL connection closed')
