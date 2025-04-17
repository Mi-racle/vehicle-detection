import json
import logging
from datetime import datetime

import mysql.connector
import yaml
from mysql.connector import Error

from db import TblGroupDAO, TblCameraDAO


class TblTaskOnlineDAO:
    def __init__(self, config_path: str):
        self.__config: dict = yaml.safe_load(open(config_path, 'r'))
        self.__connection = mysql.connector.connect(**self.__config)
        self.__TABLE_NAME = 'tbl_task_run'
        self.__TASK_INFO_DAO = TblTaskInfoDAO(config_path)
        self.__GROUP_DAO = TblGroupDAO(config_path)
        self.__CAMERA_DAO = TblCameraDAO(config_path)

    def __del__(self):
        if self.__connection and self.__connection.is_connected():
            self.__connection.close()

    def get_online_task_by_id(self, task_id: int):
        cursor = None

        try:
            if not self.__connection.is_connected():
                self.__connection = mysql.connector.connect(**self.__config)

            cursor = self.__connection.cursor(dictionary=True)

            query = f'''SELECT * FROM {self.__TABLE_NAME} WHERE id = %s'''

            cursor.execute(query, (task_id,))
            logging.info(f'Entry successfully selected from {self.__TABLE_NAME}')

            task = cursor.fetchone()

            if task:
                task['group_id'] = json.loads(task['group_id'])
                task['analysis_start_time'] = task.pop('start_time')
                task['analysis_end_time'] = task.pop('end_time')
                task['status'] = task.pop('excute_status')  # 'excute' is a typo in sql

            return task

        except Error as e:
            self.__connection.rollback()
            logging.info(f'Error: {e}')

        finally:
            if cursor:
                cursor.close()

    def get_next_online_tasks(self):
        cursor = None

        try:
            if not self.__connection.is_connected():
                self.__connection = mysql.connector.connect(**self.__config)

            cursor = self.__connection.cursor(dictionary=True)

            query = f'''SELECT * FROM {self.__TABLE_NAME} WHERE excute_status = 0'''  # 'excute' is a typo in sql

            cursor.execute(query)
            logging.info(f'Entry successfully selected from {self.__TABLE_NAME}')

            tasks = []
            fetches = cursor.fetchall()

            for fetch in fetches:
                execute_date = datetime.combine(fetch['execute_date'], datetime.min.time())
                start_datetime = execute_date + fetch['start_time']
                end_datetime = execute_date + fetch['end_time']

                if start_datetime <= datetime.now() < end_datetime:
                    task_info = self.__TASK_INFO_DAO.get_task_info_by_id(fetch['id'])
                    fetch['task_name'] = task_info['task_name'] if task_info else ''
                    fetch['group_id'] = json.loads(fetch['group_id'])
                    fetch['create_time'] = start_datetime
                    fetch['analysis_start_time'] = fetch.pop('start_time')
                    fetch['analysis_end_time'] = fetch.pop('end_time')
                    fetch['status'] = fetch.pop('excute_status')  # 'excute' is a typo in sql

                    fetch['camera_id'] = ''
                    fetch['camera_type'] = 1
                    fetch['url'] = fetch.pop('file_url') if 'file_url' in fetch else ''
                    fetch['matrix'] = ''
                    fetch['description'] = ''

                    if fetch['group_id']:
                        group = self.__GROUP_DAO.get_group_by_group_id(fetch['group_id'][0])

                        if group:
                            camera = self.__CAMERA_DAO.get_camera_by_camera_id(group['camera_id'])

                            if camera:
                                fetch['camera_id'] = camera['camera_id']
                                fetch['camera_type'] = camera['type']
                                fetch['url'] = camera['url']
                                fetch['matrix'] = camera['matrix']  # TODO
                                fetch['description'] = camera['description']

                    tasks.append(fetch)

            return tasks

        except Error as e:
            logging.info(f'Error: {e}')

        finally:
            if cursor:
                cursor.close()

    def update_online_task_status_by_id(self, task_id: int, status: int):
        """ status: -1 - exception; 0 - to be done; 1 - done """
        cursor = None

        try:
            if not self.__connection.is_connected():
                self.__connection = mysql.connector.connect(**self.__config)

            cursor = self.__connection.cursor(dictionary=True)

            update_query = f'''UPDATE {self.__TABLE_NAME} SET excute_status = %s WHERE id = %s'''  # 'excute' is a typo in sql

            cursor.execute(update_query, (status, task_id))
            self.__connection.commit()
            logging.info(f'{self.__TABLE_NAME} successfully updated')

        except Error as e:
            self.__connection.rollback()
            logging.info(f'Error: {e}')

        finally:
            if cursor:
                cursor.close()


class TblTaskInfoDAO:
    def __init__(self, config_path: str):
        self.__config: dict = yaml.safe_load(open(config_path, 'r'))
        self.__connection = mysql.connector.connect(**self.__config)
        self.__TABLE_NAME = 'tbl_task_info'

    def __del__(self):
        if self.__connection and self.__connection.is_connected():
            self.__connection.close()

    def get_task_info_by_id(self, task_id: int):
        cursor = None

        try:
            if not self.__connection.is_connected():
                self.__connection = mysql.connector.connect(**self.__config)

            cursor = self.__connection.cursor(dictionary=True)

            query = f'''SELECT * FROM {self.__TABLE_NAME} WHERE id = %s'''

            cursor.execute(query, (task_id,))
            logging.info(f'Entry successfully selected from {self.__TABLE_NAME}')

            return cursor.fetchone()

        except Error as e:
            self.__connection.rollback()
            logging.info(f'Error: {e}')

        finally:
            if cursor:
                cursor.close()
