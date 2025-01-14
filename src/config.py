import yaml

ARG_CONFIG = yaml.safe_load(open('configs/general_config.yaml', 'r'))
JAM_CONFIG = yaml.safe_load(open('configs/jam_config.yaml', 'r'))
MTP_CONFIG = yaml.safe_load(open('configs/mtp_config.yaml', 'r'))
