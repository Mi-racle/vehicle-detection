import logging

import mysql.connector
from mysql.connector import Error

from db.db_config import DB_CONFIG


def insert_result(result):
    connection = None
    cursor = None

    try:
        connection = mysql.connector.connect(**DB_CONFIG, buffered=True)
        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)

            table = 'tbl_result'
            insert_query = f"""
                INSERT INTO {table} (
                model_name, model_version, camera_type, camera_id, video_type, 
                source, dest, start_time, end_time, plate_no, locations
                ) VALUES (
                %s, %s, %s, %s, %s, 
                %s, %s, %s, %s, %s, %s
                )
            """

            result[-1] = str(result[-1])
            cursor.execute(insert_query, result)
            connection.commit()
            logging.info(f'Entry successfully inserted into {table}')

    except Error as e:
        connection.rollback()
        connection = None
        logging.error(f'Error: {e}')

    finally:
        if connection and connection.is_connected():
            if cursor:
                cursor.close()
            connection.close()
            logging.info('MySQL connection closed')


def get_result_header():
    connection = None
    cursor = None

    try:
        connection = mysql.connector.connect(**DB_CONFIG, buffered=True)
        if connection.is_connected():
            cursor = connection.cursor()

            table = 'tbl_result'
            describe_query = f'''DESCRIBE {table}'''

            cursor.execute(describe_query)
            logging.info(f'Successfully described {table}')

            return [column[0] for column in cursor.fetchall()]

    except Error as e:
        connection = None
        logging.error(f'Error: {e}')

    finally:
        if connection and connection.is_connected():
            if cursor:
                cursor.close()
            connection.close()
            logging.info('MySQL connection closed')
