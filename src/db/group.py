import json
import logging

import mysql.connector
import yaml
from mysql.connector import Error


class TblGroupDAO:
    def __init__(self, config_path: str):
        self.__config: dict = yaml.safe_load(open(config_path, 'r'))
        self.__connection = mysql.connector.connect(**self.__config)
        self._TABLE_NAME = 'tbl_group'

    def __del__(self):
        if self.__connection and self.__connection.is_connected():
            self.__connection.close()

    def get_group_by_group_id(self, group_id: int):
        cursor = None

        try:
            if not self.__connection.is_connected():
                self.__connection = mysql.connector.connect(**self.__config)

            cursor = self.__connection.cursor(dictionary=True)

            query = f'''SELECT * FROM {self._TABLE_NAME} WHERE group_id = %s'''

            cursor.execute(query, (group_id,))
            logging.info(f'Entry successfully selected from {self._TABLE_NAME}')

            group = cursor.fetchone()
            group['args'] = json.loads(group['args'])

            return group

        except Error as e:
            logging.info(f'Error: {e}')

        finally:
            if cursor:
                cursor.close()
