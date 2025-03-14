import yaml

__config__: dict = yaml.safe_load(open('configs/arg_config.yaml', 'r'))
GENERAL_CONFIG = __config__['general_args']
JAM_CONFIG = __config__['jam_args']
QUEUE_CONFIG = __config__['queue_args']
DENSITY_CONFIG = __config__['density_args']
SIZE_CONFIG = __config__['size_args']
COLOR_CONFIG = __config__['color_args']
SECTION_CONFIG = __config__['section_args']
VELOCITY_CONFIG = __config__['velocity_args']
VOLUME_CONFIG = __config__['volume_args']
PIM_CONFIG = __config__['pim_args']
PARKING_CONFIG = __config__['parking_args']
WRONGWAY_CONFIG = __config__['wrongway_args']
LANECHANGE_CONFIG = __config__['lanechange_args']
SPEEDING_CONFIG = __config__['speeding_args']
