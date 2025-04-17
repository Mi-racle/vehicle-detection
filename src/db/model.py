import json
import logging

import mysql.connector
import yaml
from mysql.connector import Error


class TblModelDAO:
    def __init__(self, config_path: str):
        self.__config: dict = yaml.safe_load(open(config_path, 'r'))
        self.__connection = mysql.connector.connect(**self.__config)
        self.__TABLE_NAME = 'tbl_model'

    def __del__(self):
        if self.__connection and self.__connection.is_connected():
            self.__connection.close()

    def get_model_by_model_id(self, model_id: int):
        cursor = None

        try:
            if not self.__connection.is_connected():
                self.__connection = mysql.connector.connect(**self.__config)

            cursor = self.__connection.cursor(dictionary=True)

            query = f'''SELECT * FROM {self.__TABLE_NAME} WHERE model_id = %s'''

            cursor.execute(query, (model_id,))
            logging.info(f'Entry successfully selected from {self.__TABLE_NAME}')

            camera = cursor.fetchone()
            camera['arg_example'] = json.loads(camera['arg_example'])

            return camera

        except Error as e:
            logging.info(f'Error: {e}')

        finally:
            if cursor:
                cursor.close()
