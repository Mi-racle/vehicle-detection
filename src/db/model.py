import json
import logging

import mysql.connector
import yaml
from mysql.connector import Error


class TblModelDAO:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.connection = mysql.connector.connect(
            **yaml.safe_load(open(self.config_path, 'r'))
        )

    def __del__(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()

    def get_model_by_model_id(self, model_id: int):
        cursor = None

        try:
            if self.connection.is_connected():
                self.connection = mysql.connector.connect(
                    **yaml.safe_load(open(self.config_path, 'r'))
                )

            cursor = self.connection.cursor(dictionary=True)

            table = 'tbl_model'
            query = f'''SELECT * FROM {table} WHERE model_id = %s'''

            cursor.execute(query, (model_id,))
            logging.info(f'Entry successfully selected from {table}')

            camera = cursor.fetchone()
            camera['arg_example'] = json.loads(camera['arg_example'])

            return camera

        except Error as e:
            logging.info(f'Error: {e}')

        finally:
            if cursor:
                cursor.close()
