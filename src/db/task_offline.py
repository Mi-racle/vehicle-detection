import logging

import mysql.connector
from mysql.connector import Error

from db.db_config import DB_CONFIG


def get_offline_task_by_id(task_id: int):
    connection = None
    cursor = None

    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)

            table = 'tbl_task_run_offline'
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


def get_next_offline_task():
    connection = None
    cursor = None

    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)

            table = 'tbl_task_run_offline'
            query = f'''SELECT * FROM {table} WHERE status = 0'''

            cursor.execute(query)
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


def update_offline_task_status_by_id(task_id: int, status: int):  # status: -1 - exception; 0 - to be done; 1 - done
    connection = None
    cursor = None

    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)

            table = 'tbl_task_run_offline'
            update_query = f'''UPDATE {table} SET status = %s WHERE id = %s'''

            cursor.execute(update_query, (status, task_id))
            connection.commit()
            logging.info(f'{table} successfully updated')

    except Error as e:
        connection = None
        logging.info(f'Error: {e}')

    finally:
        if connection and connection.is_connected():
            if cursor:
                cursor.close()
            connection.close()
            logging.info('MySQL connection closed')
