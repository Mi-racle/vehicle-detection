import json
import logging
from datetime import datetime

import mysql.connector
import yaml
from mysql.connector import Error


class TblTaskOnlineDAO:
    def __init__(self, config_path: str):
        self.__config: dict = yaml.safe_load(open(config_path, 'r'))
        self.__connection = mysql.connector.connect(**self.__config)
        self.__TABLE_NAME = 'tbl_task_run'

    def __del__(self):
        if self.__connection and self.__connection.is_connected():
            self.__connection.close()

    def get_online_task_by_id(self, task_id: int):
        cursor = None

        try:
            if self.__connection.is_connected():
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

    def get_next_online_task(self):
        cursor = None

        try:
            if self.__connection.is_connected():
                self.__connection = mysql.connector.connect(**self.__config)

            cursor = self.__connection.cursor(dictionary=True)

            query = f'''SELECT * FROM {self.__TABLE_NAME} WHERE excute_status = 0'''  # 'excute' is a typo in sql

            cursor.execute(query)
            logging.info(f'Entry successfully selected from {self.__TABLE_NAME}')

            tasks = cursor.fetchall()

            for task in tasks:
                execute_date = datetime.combine(task['execute_date'], datetime.min.time())
                start_datetime = execute_date + task['start_time']
                end_datetime = execute_date + task['end_time']

                if start_datetime <= datetime.now() < end_datetime:
                    task['group_id'] = json.loads(task['group_id'])
                    task['analysis_start_time'] = task.pop('start_time')
                    task['analysis_end_time'] = task.pop('end_time')
                    task['status'] = task.pop('excute_status')  # 'excute' is a typo in sql

                    return task

            return None

        except Error as e:
            logging.info(f'Error: {e}')

        finally:
            if cursor:
                cursor.close()

    def update_online_task_status_by_id(self, task_id: int, status: int):
        """ status: -1 - exception; 0 - to be done; 1 - done """
        cursor = None

        try:
            if self.__connection.is_connected():
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
