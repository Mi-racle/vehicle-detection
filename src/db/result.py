import logging

import mysql.connector
import yaml
from mysql.connector import Error


class TblResultDAO:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.connection = mysql.connector.connect(
            **yaml.safe_load(open(self.config_path, 'r'))
        )

    def __del__(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()

    def insert_result(self, result):
        cursor = None

        try:
            if self.connection.is_connected():
                self.connection = mysql.connector.connect(
                    **yaml.safe_load(open(self.config_path, 'r'))
                )

            cursor = self.connection.cursor(dictionary=True)

            table = 'tbl_result'
            insert_query = f"""
                INSERT INTO {table} (
                model_name, model_version, camera_type, camera_id, video_type, 
                source, dest, start_time, end_time, plate_no, locations
                ) VALUES (
                %s, %s, %s, %s, %s, 
                %s, %s, %s, %s, %s, %s
                )
            """

            result[-1] = str(result[-1])
            cursor.execute(insert_query, result)
            self.connection.commit()
            logging.info(f'Entry successfully inserted into {table}')

        except Error as e:
            self.connection.rollback()
            logging.error(f'Error: {e}')

        finally:
            if cursor:
                cursor.close()

    def get_result_header(self):
        cursor = None

        try:
            if self.connection.is_connected():
                self.connection = mysql.connector.connect(
                    **yaml.safe_load(open(self.config_path, 'r'))
                )

            cursor = self.connection.cursor()

            table = 'tbl_result'
            describe_query = f'''DESCRIBE {table}'''

            cursor.execute(describe_query)
            logging.info(f'Successfully described {table}')

            return [column[0] for column in cursor.fetchall()]

        except Error as e:
            logging.error(f'Error: {e}')

        finally:
            if cursor:
                cursor.close()
