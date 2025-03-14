import json
import logging

import mysql.connector
import yaml
from mysql.connector import Error


class TblCameraDAO:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.connection = mysql.connector.connect(
            **yaml.safe_load(open(self.config_path, 'r'))
        )
        # logging.info(f'{type(self).__name__} connection established')

    def __del__(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            # logging.info(f'{type(self).__name__} connection closed')

    def get_camera_by_camera_id(self, camera_id: str):
        cursor = None

        try:
            if self.connection.is_connected():
                self.connection = mysql.connector.connect(
                    **yaml.safe_load(open(self.config_path, 'r'))
                )

            cursor = self.connection.cursor(dictionary=True)

            table = 'tbl_camera'
            query = f'''SELECT * FROM {table} WHERE camera_id = %s'''

            cursor.execute(query, (camera_id,))
            logging.info(f'Entry successfully selected from {table}')

            camera = cursor.fetchone()
            camera['matrix'] = json.loads(camera['matrix'])

            return camera

        except Error as e:
            logging.info(f'Error: {e}')

        finally:
            if cursor:
                cursor.close()
