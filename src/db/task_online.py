import logging

import mysql.connector
import yaml
from mysql.connector import Error


class TblTaskOnlineDAO:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.connection = mysql.connector.connect(
            **yaml.safe_load(open(self.config_path, 'r'))
        )

    def __del__(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()

    def get_online_task_by_id(self, task_id: int):
        cursor = None

        try:
            if self.connection.is_connected():
                self.connection = mysql.connector.connect(
                    **yaml.safe_load(open(self.config_path, 'r'))
                )

            cursor = self.connection.cursor(dictionary=True)

            table = 'tbl_task_run'
            query = f'''SELECT * FROM {table} WHERE id = %s'''

            cursor.execute(query, (task_id,))
            logging.info(f'Entry successfully selected from {table}')

            return cursor.fetchone()

        except Error as e:
            self.connection.rollback()
            logging.info(f'Error: {e}')

        finally:
            if cursor:
                cursor.close()
