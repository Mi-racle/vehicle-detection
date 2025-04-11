from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QFont, QFontDatabase
from PyQt6.QtWidgets import QWidget, QLabel


class TaskDetailWidget(QWidget):
    def __init__(self, settings: dict, parent: Optional[QWidget] = None):
        super().__init__(parent)

        font_siyuan_cn_bold = QFont(
            QFontDatabase.applicationFontFamilies(
                QFontDatabase.addApplicationFont(settings['font_siyuan_cn_bold']))[0])
        font_siyuan_cn_medium = QFont(
            QFontDatabase.applicationFontFamilies(
                QFontDatabase.addApplicationFont(settings['font_siyuan_cn_medium']))[0])
        font_siyuan_cn_regular = QFont(
            QFontDatabase.applicationFontFamilies(
                QFontDatabase.addApplicationFont(settings['font_siyuan_cn_regular']))[0])

        self.setFixedSize(450, 655)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setObjectName('taskDetailWidget')
        self.setStyleSheet(
            f'{type(self).__name__}#{self.objectName()} {{ background-image: url({settings['background_image']}); }}')

        title_group = QWidget(self)
        title_group.setGeometry(16, 16, 418, 44)

        # title_group BEGIN
        self.__curr_task_icon_label = QLabel(title_group)
        self.__curr_task_icon_label.setGeometry(8, 4, 22, 22)
        self.__curr_task_icon_label.setPixmap(QPixmap(settings['curr_task_icon']))

        self.__curr_task_tag_label = QLabel('当前任务', title_group)
        self.__curr_task_tag_label.setGeometry(38, 4, 100, 22)
        self.__curr_task_tag_label.setFont(font_siyuan_cn_bold)
        self.__curr_task_tag_label.setStyleSheet(settings['curr_task_tag_label_ss'])

        self.__curr_task_line_label = QLabel(title_group)
        self.__curr_task_line_label.setGeometry(8, 41, 402, 3)
        self.__curr_task_line_label.setPixmap(QPixmap(settings['curr_task_line']))
        # title_group END

        self.__ai_icon_label = QLabel(self)
        self.__ai_icon_label.setGeometry(32, 76, 386, 168)
        self.__ai_icon_label.setPixmap(QPixmap(settings['ai_icon']))

        task_name_group = QWidget(self)
        task_name_group.setGeometry(32, 263, 398, 18)

        # task_name_group BEGIN
        self.__point_icon_label = QLabel(task_name_group)
        self.__point_icon_label.setGeometry(0, 6, 6, 6)
        self.__point_icon_label.setPixmap(QPixmap(settings['point_icon']))

        self.__task_name_label = QLabel('等待任务中', task_name_group)
        self.__task_name_label.setGeometry(14, 0, 372, 18)
        self.__task_name_label.setFont(font_siyuan_cn_medium)
        self.__task_name_label.setStyleSheet(settings['task_name_label_ss'])
        # task_name_group END

        camera_position_group = QWidget(self)
        camera_position_group.setGeometry(32, 305, 188, 48)

        # camera_position_group BEGIN
        self.__camera_position_icon_label = QLabel(camera_position_group)
        self.__camera_position_icon_label.setGeometry(0, 0, 48, 48)
        self.__camera_position_icon_label.setPixmap(QPixmap(settings['camera_position_icon']))

        camera_position_subgroup = QWidget(camera_position_group)
        camera_position_subgroup.setGeometry(64, 3, 124, 44)

        # camera_position_subgroup BEGIN
        self.__camera_position_tag_label = QLabel('摄像头位置', camera_position_subgroup)
        self.__camera_position_tag_label.setGeometry(0, 0, 124, 16)
        self.__camera_position_tag_label.setFont(font_siyuan_cn_regular)
        self.__camera_position_tag_label.setStyleSheet(settings['camera_position_tag_label_ss'])

        self.__camera_position_label = QLabel('-', camera_position_subgroup)
        self.__camera_position_label.setGeometry(0, 26, 124, 18)
        self.__camera_position_label.setFont(font_siyuan_cn_regular)
        self.__camera_position_label.setStyleSheet(settings['camera_position_label_ss'])
        # camera_position_subgroup END
        # camera_position_group END

        video_source_group = QWidget(self)
        video_source_group.setGeometry(230, 305, 188, 48)

        # video_source_group BEGIN
        self.__video_source_icon_label = QLabel(video_source_group)
        self.__video_source_icon_label.setGeometry(0, 0, 48, 48)
        self.__video_source_icon_label.setPixmap(QPixmap(settings['video_source_icon']))

        video_source_subgroup = QWidget(video_source_group)
        video_source_subgroup.setGeometry(64, 3, 124, 44)

        # video_source_subgroup BEGIN
        self.__video_source_tag_label = QLabel('视频源', video_source_subgroup)
        self.__video_source_tag_label.setGeometry(0, 0, 124, 16)
        self.__video_source_tag_label.setFont(font_siyuan_cn_regular)
        self.__video_source_tag_label.setStyleSheet(settings['video_source_tag_label_ss'])

        self.__video_source_label = QLabel('-', video_source_subgroup)
        self.__video_source_label.setGeometry(0, 26, 124, 18)
        self.__video_source_label.setFont(font_siyuan_cn_regular)
        self.__video_source_label.setStyleSheet(settings['video_source_label_ss'])
        # video_source_subgroup END
        # video_source_group END

        detection_group = QWidget(self)
        detection_group.setGeometry(32, 387, 386, 103)

        # detection_group BEGIN
        detection_tag_group = QWidget(detection_group)
        detection_tag_group.setGeometry(0, 0, 86, 18)

        # detection_tag_group BEGIN
        self.__point_icon2_label = QLabel(detection_tag_group)
        self.__point_icon2_label.setGeometry(0, 6, 6, 6)
        self.__point_icon2_label.setPixmap(QPixmap(settings['point2_icon']))

        self.__detection_tag_label = QLabel('识别内容', detection_tag_group)
        self.__detection_tag_label.setGeometry(14, 0, 72, 18)
        self.__detection_tag_label.setFont(font_siyuan_cn_medium)
        self.__detection_tag_label.setStyleSheet(settings['detection_tag_label_ss'])
        # detection_tag_group END

        self.__detection_background = QWidget(detection_group)
        self.__detection_background.setGeometry(0, 33, 386, 70)
        self.__detection_background.setObjectName('detectionBackground')
        self.__detection_background.setStyleSheet(
            f'QWidget#{self.__detection_background.objectName()} {{ {settings['detection_background_ss']} }}')

        self.__detection_name_label = QLabel('-', detection_group)  # TODO need add scroll
        self.__detection_name_label.setGeometry(132, 44, 122, 48)
        self.__detection_name_label.setFont(font_siyuan_cn_regular)
        self.__detection_name_label.setStyleSheet(settings['detection_name_label_ss'])
        # detection_group END

        time_group = QWidget(self)
        time_group.setGeometry(32, 524, 386, 73)

        # time_group BEGIN
        start_time_group = QWidget(time_group)
        start_time_group.setGeometry(0, 0, 170, 47)

        # start_time_group BEGIN
        self.__start_time_tag_label = QLabel('开始时间', start_time_group)
        self.__start_time_tag_label.setGeometry(0, 0, 64, 16)
        self.__start_time_tag_label.setFont(font_siyuan_cn_regular)
        self.__start_time_tag_label.setStyleSheet(settings['start_time_tag_label_ss'])

        self.__start_time_label = QLabel('-', start_time_group)
        self.__start_time_label.setGeometry(0, 29, 170, 18)
        self.__start_time_label.setFont(font_siyuan_cn_regular)
        self.__start_time_label.setStyleSheet(settings['start_time_label_ss'])
        # start_time_group END

        end_time_group = QWidget(time_group)
        end_time_group.setGeometry(203, 0, 170, 47)

        # end_time_group BEGIN
        self.__end_time_tag_label = QLabel('结束时间', end_time_group)
        self.__end_time_tag_label.setGeometry(0, 0, 64, 16)
        self.__end_time_tag_label.setFont(font_siyuan_cn_regular)
        self.__end_time_tag_label.setStyleSheet(settings['end_time_tag_label_ss'])

        self.__end_time_label = QLabel('-', end_time_group)
        self.__end_time_label.setGeometry(0, 29, 170, 18)
        self.__end_time_label.setFont(font_siyuan_cn_regular)
        self.__end_time_label.setStyleSheet(settings['end_time_label_ss'])
        # end_time_group END

        self.__dashed_line = QWidget(time_group)
        self.__dashed_line.setGeometry(0, 72, 386, 1)
        self.__dashed_line.setObjectName('dashedLine')
        self.__dashed_line.setStyleSheet(
            f'QWidget#{self.__dashed_line.objectName()} {{ {settings['dashed_line_ss']} }}')
        # time_group END

        creation_time_group = QWidget(self)
        creation_time_group.setGeometry(32, 615, 386, 16)

        # creation_time_group BEGIN
        self.__creation_time_tag_label = QLabel('创建时间', creation_time_group)
        self.__creation_time_tag_label.setGeometry(0, 0, 64, 16)
        self.__creation_time_tag_label.setFont(font_siyuan_cn_regular)
        self.__creation_time_tag_label.setStyleSheet(settings['creation_time_tag_label_ss'])

        self.__creation_time_label = QLabel('-', creation_time_group)
        self.__creation_time_label.setGeometry(74, 0, 312, 16)
        self.__creation_time_label.setFont(font_siyuan_cn_regular)
        self.__creation_time_label.setStyleSheet(settings['creation_time_label_ss'])
        # creation_time_group END

    def set_task(self, task_entry: dict):
        self.__task_name_label.setText(task_entry['task_name'])
        self.__camera_position_label.setText('TBD')  # description TODO
        self.__video_source_label.setText('TBD')  # url TODO
        self.__creation_time_label.setText(str(task_entry['create_time']))
        self.__start_time_label.setText(str(task_entry['analysis_start_time']))
        self.__end_time_label.setText(str(task_entry['analysis_end_time']))
        self.__detection_name_label.setText(str(task_entry['group_id']))  # TODO
