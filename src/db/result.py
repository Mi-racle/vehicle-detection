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

    def insert_result(self, result):
        cursor = None

        try:
            if self.__connection.is_connected():
                self.__connection = mysql.connector.connect(**self.__config)

            cursor = self.__connection.cursor(dictionary=True)

            insert_query = f"""
                INSERT INTO {self.__TABLE_NAME} (
                model_name, model_version, camera_type, camera_id, video_type, 
                source, dest, start_time, end_time, plate_no, locations
                ) VALUES (
                %s, %s, %s, %s, %s, 
                %s, %s, %s, %s, %s, %s
                )
            """

            result[-1] = str(result[-1])
            cursor.execute(insert_query, result)
            self.__connection.commit()
            logging.info(f'Entry successfully inserted into {self.__TABLE_NAME}')

        except Error as e:
            self.__connection.rollback()
            logging.error(f'Error: {e}')

        finally:
            if cursor:
                cursor.close()

    def get_result_header(self):
        cursor = None

        try:
            if self.__connection.is_connected():
                self.__connection = mysql.connector.connect(**self.__config)

            cursor = self.__connection.cursor()

            describe_query = f'''DESCRIBE {self.__TABLE_NAME}'''

            cursor.execute(describe_query)
            logging.info(f'Successfully described {self.__TABLE_NAME}')

            return [column[0] for column in cursor.fetchall()]

        except Error as e:
            logging.error(f'Error: {e}')

        finally:
            if cursor:
                cursor.close()
