import yaml
from obs import ObsClient


class ObsDAO:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.obs_config: dict = yaml.safe_load(open(self.config_path, 'r'))
        self.obs_client = ObsClient(
            access_key_id=self.obs_config['access_key'],
            secret_access_key=self.obs_config['access_secret'],
            server=f'http://{self.obs_config['address']}',
            port=self.obs_config['port']
        )

    def put_file(self, string: str):
        # TODO
        pass
