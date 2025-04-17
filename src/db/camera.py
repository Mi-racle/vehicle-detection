import json
import logging

import mysql.connector
import yaml
from mysql.connector import Error


class TblCameraDAO:
    def __init__(self, config_path: str):
        self.__config: dict = yaml.safe_load(open(config_path, 'r'))
        self.__connection = mysql.connector.connect(**self.__config)
        self.__TABLE_NAME = 'tbl_camera'
        # logging.info(f'{type(self).__name__} connection established')

    def __del__(self):
        if self.__connection and self.__connection.is_connected():
            self.__connection.close()
            # logging.info(f'{type(self).__name__} connection closed')

    def get_camera_by_camera_id(self, camera_id: str):
        cursor = None

        try:
            if not self.__connection.is_connected():
                self.__connection = mysql.connector.connect(**self.__config)

            cursor = self.__connection.cursor(dictionary=True)

            query = f'''SELECT * FROM {self.__TABLE_NAME} WHERE camera_id = %s'''

            cursor.execute(query, (camera_id,))
            logging.info(f'Entry successfully selected from {self.__TABLE_NAME}')

            camera = cursor.fetchone()
            camera['matrix'] = json.loads(camera['matrix'])

            return camera

        except Error as e:
            logging.info(f'Error: {e}')

        finally:
            if cursor:
                cursor.close()
