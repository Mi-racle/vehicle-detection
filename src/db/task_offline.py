import json
import logging

import mysql.connector
import yaml
from mysql.connector import Error


class TblTaskOfflineDAO:
    def __init__(self, config_path: str):
        self.__config: dict = yaml.safe_load(open(config_path, 'r'))
        self.__connection = mysql.connector.connect(**self.__config)
        self.__TABLE_NAME = 'tbl_task_run_offline'

    def __del__(self):
        if self.__connection and self.__connection.is_connected():
            self.__connection.close()

    def get_offline_task_by_id(self, task_id: int):
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

            return task

        except Error as e:
            self.__connection.rollback()
            logging.info(f'Error: {e}')

        finally:
            if cursor:
                cursor.close()

    def get_next_offline_task(self):
        cursor = None

        try:
            if self.__connection.is_connected():
                self.__connection = mysql.connector.connect(**self.__config)

            cursor = self.__connection.cursor(dictionary=True)

            query = f'''SELECT * FROM {self.__TABLE_NAME} WHERE status = 0'''

            cursor.execute(query)
            logging.info(f'Entry successfully selected from {self.__TABLE_NAME}')

            tasks = []
            fetches = cursor.fetchall()

            for fetch in fetches:
                fetch['task_name'] = fetch.pop('offline_task_name')
                fetch['group_id'] = json.loads(fetch['group_id'])
                tasks.append(fetch)

            return tasks

        except Error as e:
            logging.info(f'Error: {e}')

        finally:
            if cursor:
                cursor.close()

    def update_offline_task_status_by_id(self, task_id: int, status: int):
        """ status: -1 - exception; 0 - to be done; 1 - done """
        cursor = None

        try:
            if self.__connection.is_connected():
                self.__connection = mysql.connector.connect(**self.__config)

            cursor = self.__connection.cursor(dictionary=True)

            update_query = f'''UPDATE {self.__TABLE_NAME} SET status = %s WHERE id = %s'''

            cursor.execute(update_query, (status, task_id))
            self.__connection.commit()
            logging.info(f'{self.__TABLE_NAME} successfully updated')

        except Error as e:
            self.__connection.rollback()
            logging.info(f'Error: {e}')

        finally:
            if cursor:
                cursor.close()
