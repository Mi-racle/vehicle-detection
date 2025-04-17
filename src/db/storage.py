import logging
import os

import yaml
from obs import ObsClient


class ObsDAO:
    def __init__(self, config_path: str):
        self.__obs_config: dict = yaml.safe_load(open(config_path, 'r'))
        self.__obs_client = ObsClient(
            access_key_id=self.__obs_config['access_key'],
            secret_access_key=self.__obs_config['access_secret'],
            server=f'http://{self.__obs_config['address']}',
            port=self.__obs_config['port']
        )

    def upload_file(self, file_path: str, object_key: str | None = None):
        resp = None

        try:
            resp = self.__obs_client.uploadFile(
                bucketName=self.__obs_config['bucket'],
                objectKey=object_key if object_key else os.path.basename(file_path),
                uploadFile=file_path
            )

        except Exception as e:
            logging.error(e)

        finally:
            if resp:
                if resp.status < 300:
                    logging.info(f'{file_path} uploaded successfully. ETag: {resp.body.etag}')
                else:
                    logging.info(f'Failed to upload {file_path}. Reason: {resp.reason}. Status: {resp.status}')

            else:
                logging.error(f'Failed to upload {file_path}. Reason: unknown. Status: unknown')

    def download_file(self, download_path: str, object_key: str | None = None):
        resp = None
        key = object_key if object_key else os.path.basename(download_path)

        try:
            resp = self.__obs_client.getObject(
                bucketName=self.__obs_config['bucket'],
                objectKey=key,
                downloadPath=download_path
            )

        except Exception as e:
            logging.error(e)

        finally:
            if resp:
                if resp.status < 300:
                    logging.info(f'{key} downloaded successfully to {download_path}. ETag: {resp.body.etag}')
                else:
                    logging.info(f'Failed to download {key}. Reason: {resp.reason}. Status: {resp.status}')

            else:
                logging.error(f'Failed to download {key}. Reason: unknown. Status: unknown')

    def delete_file(self, object_key: str):
        resp = None

        try:
            resp = self.__obs_client.deleteObject(
                bucketName=self.__obs_config['bucket'],
                objectKey=object_key
            )

        except Exception as e:
            logging.error(e)

        finally:
            if resp:
                if resp.status < 300:
                    logging.info(f'{object_key} deleted successfully. ETag: {resp.body.etag}')
                else:
                    logging.info(f'Failed to delete {object_key}. Reason: {resp.reason}. Status: {resp.status}')

            else:
                logging.error(f'Failed to delete {object_key}. Reason: unknown. Status: unknown')

    def list_buckets(self):
        resp = self.__obs_client.listBuckets()

        if resp.status < 300:
            print(f'Buckets:', [obj_dict['name'] for obj_dict in resp['body']['buckets']])

        else:
            print(f'Error in list_buckets: {resp.status}')

    def list_objects(self):
        resp = self.__obs_client.listObjects(bucketName=self.__obs_config['bucket'])

        if resp.status < 300:
            print(f'Objects in bucket \'{self.__obs_config['bucket']}\': ')

            for obj_dict in resp['body']['contents']:
                print(obj_dict['key'])

        else:
            print(f'Error in list_objects: {resp.status}')


if __name__ == '__main__':
    obs_dao = ObsDAO('../configs/obs_config.yaml')
    obs_dao.list_buckets()
    obs_dao.list_objects()
    # obs_dao.upload_file('test.txt')
    # obs_dao.download_file('test.txt')
    # obs_dao.delete_file('test2.txt')
    # obs_dao.list_objects()
