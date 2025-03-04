import json
import logging

import mysql.connector
from mysql.connector import Error

from db.db_config import DB_CONFIG


def get_model_by_model_id(model_id: int):
    connection = None
    cursor = None

    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)

            table = 'tbl_model'
            query = f'''SELECT * FROM {table} WHERE model_id = %s'''

            cursor.execute(query, (model_id,))
            logging.info(f'Entry successfully selected from {table}')

            camera = cursor.fetchone()
            camera['example'] = json.loads(camera['arg_example'])

            return camera

    except Error as e:
        connection = None
        logging.info(f'Error: {e}')

    finally:
        if connection and connection.is_connected():
            if cursor:
                cursor.close()
            connection.close()
            logging.info('MySQL connection closed')
