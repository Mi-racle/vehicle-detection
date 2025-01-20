import yaml

__config__: dict = yaml.safe_load(open('configs/arg_config.yaml', 'r'))
GENERAL_CONFIG = __config__['general_args']
JAM_CONFIG = __config__['jam_args']
MIP_CONFIG = __config__['mip_args']
SIZE_CONFIG = __config__['size_args']
QUEUE_CONFIG = __config__['queue_args']
DENSITY_CONFIG = __config__['density_args']
VOLUME_CONFIG = __config__['volume_args']
SPEED_CONFIG = __config__['speed_args']
