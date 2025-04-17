import logging

import mysql.connector
import yaml
from mysql.connector import Error


class TblResultDAO:
    def __init__(self, config_path: str):
        self.__config: dict = yaml.safe_load(open(config_path, 'r'))
        self.__connection = mysql.connector.connect(**self.__config)
        self.__TABLE_NAME = 'tbl_result'

    def __del__(self):
        if self.__connection and self.__connection.is_connected():
            self.__connection.close()

    def insert_result(self, result: dict):
        cursor = None

        try:
            if not self.__connection.is_connected():
                self.__connection = mysql.connector.connect(**self.__config)

            cursor = self.__connection.cursor(dictionary=True)

            cols = ', '.join(result.keys())
            placeholders = ', '.join(['%s'] * len(result))
            insert_query = f'INSERT INTO {self.__TABLE_NAME} ({cols}) VALUES ({placeholders})'

            result['locations'] = str(result['locations'])
            cursor.execute(insert_query, tuple(result.values()))
            self.__connection.commit()
            logging.info(f'Entry successfully inserted into {self.__TABLE_NAME}')

        except Error as e:
            self.__connection.rollback()
            logging.error(f'Error: {e}')

        finally:
            if cursor:
                cursor.close()
