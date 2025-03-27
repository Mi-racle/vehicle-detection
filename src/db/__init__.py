from db.camera import TblCameraDAO
from db.group import TblGroupDAO
from db.model import TblModelDAO
from db.result import TblResultDAO
from db.storage import ObsDAO
from db.task_offline import TblTaskOfflineDAO
from db.task_online import TblTaskOnlineDAO

DB_CONFIG_PATH = 'configs/db_config.yaml'
CAMERA_DAO = TblCameraDAO(DB_CONFIG_PATH)
GROUP_DAO = TblGroupDAO(DB_CONFIG_PATH)
MODEL_DAO = TblModelDAO(DB_CONFIG_PATH)
RESULT_DAO = TblResultDAO(DB_CONFIG_PATH)
TASK_OFFLINE_DAO = TblTaskOfflineDAO(DB_CONFIG_PATH)
TASK_ONLINE_DAO = TblTaskOnlineDAO(DB_CONFIG_PATH)

OBS_CONFIG_PATH = 'configs/obs_config.yaml'
OBS_DAO = ObsDAO(OBS_CONFIG_PATH)
