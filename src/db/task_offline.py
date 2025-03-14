import json
import logging

import mysql.connector
import yaml
from mysql.connector import Error


class TblTaskOfflineDAO:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.connection = mysql.connector.connect(
            **yaml.safe_load(open(self.config_path, 'r'))
        )

    def __del__(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()

    def get_offline_task_by_id(self, task_id: int):
        cursor = None

        try:
            if self.connection.is_connected():
                self.connection = mysql.connector.connect(
                    **yaml.safe_load(open(self.config_path, 'r'))
                )

            cursor = self.connection.cursor(dictionary=True)

            table = 'tbl_task_run_offline'
            query = f'''SELECT * FROM {table} WHERE id = %s'''

            cursor.execute(query, (task_id,))
            logging.info(f'Entry successfully selected from {table}')

            task = cursor.fetchone()
            task['group_id'] = json.loads(task['group_id'])

            return task

        except Error as e:
            logging.info(f'Error: {e}')

        finally:
            if cursor:
                cursor.close()

    def get_next_offline_task(self):
        cursor = None

        try:
            if self.connection.is_connected():
                self.connection = mysql.connector.connect(
                    **yaml.safe_load(open(self.config_path, 'r'))
                )

            cursor = self.connection.cursor(dictionary=True)

            table = 'tbl_task_run_offline'
            query = f'''SELECT * FROM {table} WHERE status = 0'''

            cursor.execute(query)
            logging.info(f'Entry successfully selected from {table}')

            task = cursor.fetchone()
            task['group_id'] = json.loads(task['group_id'])

            return task

        except Error as e:
            logging.info(f'Error: {e}')

        finally:
            if cursor:
                cursor.close()

    ''' status: -1 - exception; 0 - to be done; 1 - done '''
    def update_offline_task_status_by_id(self, task_id: int, status: int):
        cursor = None

        try:
            if self.connection.is_connected():
                self.connection = mysql.connector.connect(
                    **yaml.safe_load(open(self.config_path, 'r'))
                )

            cursor = self.connection.cursor(dictionary=True)

            table = 'tbl_task_run_offline'
            update_query = f'''UPDATE {table} SET status = %s WHERE id = %s'''

            cursor.execute(update_query, (status, task_id))
            self.connection.commit()
            logging.info(f'{table} successfully updated')

        except Error as e:
            self.connection.rollback()
            logging.info(f'Error: {e}')

        finally:
            if cursor:
                cursor.close()
