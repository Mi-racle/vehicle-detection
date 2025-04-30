import logging

import mysql.connector
import yaml
from mysql.connector import Error


class SysconDAO:
    def __init__(self, config_path: str):
        self.__config: dict = yaml.safe_load(open(config_path, 'r'))
        self.__connection = mysql.connector.connect(**self.__config)
        self._TABLE_NAME = 'sys_config'

    def __del__(self):
        if self.__connection and self.__connection.is_connected():
            self.__connection.close()

    def get_url_prefix(self):
        cursor = None
        column_name = 'config_value'
        config_key = 'httpUrl'

        try:
            if not self.__connection.is_connected():
                self.__connection = mysql.connector.connect(**self.__config)

            cursor = self.__connection.cursor(dictionary=True)

            query = f'''SELECT {column_name} FROM {self._TABLE_NAME} WHERE config_key = %s'''

            cursor.execute(query, (config_key, ))
            logging.info(f'Entry successfully selected from {self._TABLE_NAME}')

            config_key = cursor.fetchone()[column_name]

            return config_key

        except Error as e:
            logging.info(f'Error: {e}')

        finally:
            if cursor:
                cursor.close()
