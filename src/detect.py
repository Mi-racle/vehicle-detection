from ultralytics.engine.results import Results

from config import JAM_CONFIG


def detect_jam(result: Results) -> bool:
    target_indices = JAM_CONFIG['cls_indices']
    threshold = JAM_CONFIG['threshold']

    count = 0
    for ele in result.boxes.cls:
        if ele in target_indices:
            count += 1

            if count >= threshold:
                return True

    return False


def detect_motor_into_pavement(result: Results) -> bool:
    return True
