import json
import logging

import mysql.connector
from mysql.connector import Error

from db.db_config import DB_CONFIG


def get_camera_by_camera_id(camera_id: str):
    connection = None
    cursor = None

    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)

            table = 'tbl_camera'
            query = f'''SELECT * FROM {table} WHERE camera_id = %s'''

            cursor.execute(query, (camera_id,))
            logging.info(f'Entry successfully selected from {table}')

            camera = cursor.fetchone()
            camera['matrix'] = json.loads(camera['matrix'])

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
