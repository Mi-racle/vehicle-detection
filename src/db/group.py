import json
import logging

import mysql.connector
import yaml
from mysql.connector import Error


class TblGroupDAO:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.connection = mysql.connector.connect(
            **yaml.safe_load(open(self.config_path, 'r'))
        )

    def __del__(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()

    def get_group_by_group_id(self, group_id: int):
        cursor = None

        try:
            if self.connection.is_connected():
                self.connection = mysql.connector.connect(
                    **yaml.safe_load(open(self.config_path, 'r'))
                )

            cursor = self.connection.cursor(dictionary=True)

            table = 'tbl_group'
            query = f'''SELECT * FROM {table} WHERE group_id = %s'''

            cursor.execute(query, (group_id,))
            logging.info(f'Entry successfully selected from {table}')

            group = cursor.fetchone()
            group['args'] = json.loads(group['args'])

            return group

        except Error as e:
            logging.info(f'Error: {e}')

        finally:
            if cursor:
                cursor.close()
