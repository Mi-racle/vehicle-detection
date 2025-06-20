from datetime import datetime, timedelta
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QFont, QFontDatabase, QResizeEvent
from PyQt6.QtWidgets import QWidget, QLabel, QScrollArea, QVBoxLayout

from db import GROUP_DAO, MODEL_DAO
from ui.ui_utils import ScrollContainer
from utils import get_video_seconds


class TaskDetailWidget(QWidget):
    add_detection_signal = pyqtSignal(str)
    reset_detection_signal = pyqtSignal()

    def __init__(self, settings: dict, parent: Optional[QWidget] = None):
        super().__init__(parent)

        font_families = QFontDatabase.families()
        font_siyuan_cn_bold = (
            QFont('Source Han Sans CN Bold')) if 'Source Han Sans CN Bold' in font_families else QFont(
                QFontDatabase.applicationFontFamilies(
                    QFontDatabase.addApplicationFont(settings['font_siyuan_cn_bold']))[0])
        font_siyuan_cn_medium = (
            QFont('Source Han Sans CN Medium')) if 'Source Han Sans CN Medium' in font_families else QFont(
                QFontDatabase.applicationFontFamilies(
                    QFontDatabase.addApplicationFont(settings['font_siyuan_cn_medium']))[0])
        font_siyuan_cn_regular = (
            QFont('Source Han Sans CN')) if 'Source Han Sans CN' in font_families else QFont(
                QFontDatabase.applicationFontFamilies(
                    QFontDatabase.addApplicationFont(settings['font_siyuan_cn_regular']))[0])

        self.__background = QLabel(self)
        self.__background.setGeometry(0, 0, 450, 655)
        self.__background.setPixmap(QPixmap(settings['background_image']))
        self.__background.setScaledContents(True)

        title_group = QWidget()
        title_group.setFixedSize(418, 26)

        # title_group BEGIN
        self.__curr_task_icon_label = QLabel(title_group)
        self.__curr_task_icon_label.setGeometry(8, 4, 22, 22)
        self.__curr_task_icon_label.setPixmap(QPixmap(settings['curr_task_icon']))

        self.__curr_task_tag_label = QLabel('当前任务', title_group)
        self.__curr_task_tag_label.setGeometry(38, 4, 100, 22)
        self.__curr_task_tag_label.setFont(font_siyuan_cn_bold)
        self.__curr_task_tag_label.setStyleSheet(settings['curr_task_tag_label_ss'])
        # title_group END

        line_group = QWidget()
        line_group.setFixedSize(418, 3)

        # line_group BEGIN
        self.__curr_task_line_label = QLabel(line_group)
        self.__curr_task_line_label.setGeometry(8, 0, 402, 3)
        self.__curr_task_line_label.setPixmap(QPixmap(settings['curr_task_line']))
        # line_group END

        ai_icon_group = QWidget()
        ai_icon_group.setFixedSize(418, 168)

        # ai_icon_group BEGIN
        self.__ai_icon_label = QLabel(ai_icon_group)
        self.__ai_icon_label.setGeometry(16, 0, 386, 168)
        self.__ai_icon_label.setPixmap(QPixmap(settings['ai_icon']))
        # ai_icon_group END

        task_name_group = QWidget()
        task_name_group.setFixedSize(418, 18)

        # task_name_group BEGIN
        self.__point_icon_label = QLabel(task_name_group)
        self.__point_icon_label.setGeometry(16, 6, 6, 6)
        self.__point_icon_label.setPixmap(QPixmap(settings['point_icon']))

        self.__task_name_label = QLabel('等待任务中', task_name_group)
        self.__task_name_label.setGeometry(30, 0, 372, 18)
        self.__task_name_label.setFont(font_siyuan_cn_medium)
        self.__task_name_label.setStyleSheet(settings['task_name_label_ss'])
        # task_name_group END

        pos_src_group = QWidget()
        pos_src_group.setFixedSize(418, 48)

        # pos_src_group BEGIN
        camera_position_group = QWidget(pos_src_group)
        camera_position_group.setGeometry(16, 0, 188, 48)

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
        self.__camera_position_label.setToolTip(self.__camera_position_label.text())
        self.__camera_position_label.setFont(font_siyuan_cn_regular)
        self.__camera_position_label.setStyleSheet(settings['camera_position_label_ss'])
        # camera_position_subgroup END
        # camera_position_group END

        video_source_group = QWidget(pos_src_group)
        video_source_group.setGeometry(214, 0, 188, 48)

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
        self.__video_source_label.setToolTip(self.__video_source_label.text())
        self.__video_source_label.setFont(font_siyuan_cn_regular)
        self.__video_source_label.setStyleSheet(settings['video_source_label_ss'])
        # video_source_subgroup END
        # video_source_group END
        # pos_src_group END

        detection_tag_group = QWidget()
        detection_tag_group.setFixedSize(418, 18)

        # detection_tag_group BEGIN
        self.__point_icon2_label = QLabel(detection_tag_group)
        self.__point_icon2_label.setGeometry(16, 6, 6, 6)
        self.__point_icon2_label.setPixmap(QPixmap(settings['point2_icon']))

        self.__detection_tag_label = QLabel('识别模型', detection_tag_group)
        self.__detection_tag_label.setGeometry(30, 0, 72, 18)
        self.__detection_tag_label.setFont(font_siyuan_cn_medium)
        self.__detection_tag_label.setStyleSheet(settings['detection_tag_label_ss'])
        # detection_tag_group END

        detection_group = QWidget()
        detection_group.setFixedSize(418, 70)

        # detection_group BEGIN
        self.__detection_background = QWidget(detection_group)
        self.__detection_background.setGeometry(16, 0, 386, 70)
        self.__detection_background.setObjectName('detectionBackground')
        self.__detection_background.setStyleSheet(
            f'QWidget#{self.__detection_background.objectName()} {{ {settings['detection_background_ss']} }}')

        self.__scroll_container = ScrollContainer()
        self.__scroll_container.setFixedSize(0, 0)

        self.__scroll_area = QScrollArea(detection_group)
        self.__scroll_area.setGeometry(18, 11, 332 + 24, 48)  # 24 = 14+8+1*2 = gap + bar_width + bar_border * 2
        self.__scroll_area.setWidget(self.__scroll_container)
        self.__scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self.__scroll_area.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.__scroll_area.setStyleSheet(f'''
                    QScrollArea {{ background: transparent;}}
                    {settings['scroll_bar_ss']}
                    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; background: none;}}
                    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none;}}
                ''')

        self.__scroll_item_font = font_siyuan_cn_regular
        self.__scroll_item_ss = settings['detection_name_label_ss']

        time_group = QWidget()
        time_group.setFixedSize(418, 47)

        # time_group BEGIN
        start_time_group = QWidget(time_group)
        start_time_group.setGeometry(16, 0, 170, 47)

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
        end_time_group.setGeometry(219, 0, 170, 47)

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
        # time_group END

        dashed_line_group = QWidget()
        dashed_line_group.setFixedSize(418, 1)

        # dashed_line_group BEGIN
        self.__dashed_line = QWidget(dashed_line_group)
        self.__dashed_line.setGeometry(16, 0, 386, 1)
        self.__dashed_line.setObjectName('dashedLine')
        self.__dashed_line.setStyleSheet(
            f'QWidget#{self.__dashed_line.objectName()} {{ {settings['dashed_line_ss']} }}')
        # dashed_line_group END

        creation_time_group = QWidget()
        creation_time_group.setFixedSize(418, 16)

        # creation_time_group BEGIN
        self.__creation_time_tag_label = QLabel('创建时间', creation_time_group)
        self.__creation_time_tag_label.setGeometry(16, 0, 64, 16)
        self.__creation_time_tag_label.setFont(font_siyuan_cn_regular)
        self.__creation_time_tag_label.setStyleSheet(settings['creation_time_tag_label_ss'])

        self.__creation_time_label = QLabel('-', creation_time_group)
        self.__creation_time_label.setGeometry(90, 0, 312, 16)
        self.__creation_time_label.setFont(font_siyuan_cn_regular)
        self.__creation_time_label.setStyleSheet(settings['creation_time_label_ss'])
        # creation_time_group END

        vertical_layout = QVBoxLayout(self)
        vertical_layout.setSpacing(0)
        vertical_layout.setContentsMargins(16, 16, 16, 24)

        vertical_layout.addWidget(title_group)
        vertical_layout.addStretch(15)
        vertical_layout.addWidget(line_group)
        vertical_layout.addStretch(16)
        vertical_layout.addWidget(ai_icon_group)
        vertical_layout.addStretch(19)
        vertical_layout.addWidget(task_name_group)
        vertical_layout.addStretch(24)
        vertical_layout.addWidget(pos_src_group)
        vertical_layout.addStretch(34)
        vertical_layout.addWidget(detection_tag_group)
        vertical_layout.addStretch(15)
        vertical_layout.addWidget(detection_group)
        vertical_layout.addStretch(34)
        vertical_layout.addWidget(time_group)
        vertical_layout.addStretch(25)
        vertical_layout.addWidget(dashed_line_group)
        vertical_layout.addStretch(18)
        vertical_layout.addWidget(creation_time_group)

    def resizeEvent(self, event: QResizeEvent):
        if event.oldSize().width() != -1 and event.oldSize().height() != -1:
            self.__background.resize(event.size())

        super().resizeEvent(event)

    def set_task(self, task_entry: dict | None = None):
        if task_entry:
            self.__task_name_label.setText(task_entry['task_name'])
            self.__camera_position_label.setText(task_entry['description'])
            self.__camera_position_label.setToolTip(self.__camera_position_label.text())
            self.__video_source_label.setText(task_entry.get('download_url') or task_entry['url'])
            self.__video_source_label.setToolTip(self.__video_source_label.text())
            self.__creation_time_label.setText(str(task_entry['create_time']))
            start_time = task_entry.get('analysis_start_time') or timedelta()
            end_time = task_entry.get('analysis_end_time') or timedelta(seconds=get_video_seconds(task_entry['url']))
            if 'execute_date' in task_entry:
                start_time = datetime.combine(task_entry['execute_date'], datetime.min.time()) + start_time
                end_time = datetime.combine(task_entry['execute_date'], datetime.min.time()) + end_time
            self.__start_time_label.setText(str(start_time))
            self.__end_time_label.setText(str(end_time))
            for group_id in task_entry['group_id']:
                group = GROUP_DAO.get_group_by_group_id(group_id)
                if not group:
                    continue
                model = MODEL_DAO.get_model_by_model_id(group['model_id'])
                detection = f'{model['model_name']}v{model['model_version']}'
                self.add_detection_signal.emit(detection)

        else:  # reset
            self.__task_name_label.setText('等待任务中')
            self.__camera_position_label.setText('-')
            self.__camera_position_label.setToolTip(self.__camera_position_label.text())
            self.__video_source_label.setText('-')
            self.__video_source_label.setToolTip(self.__video_source_label.text())
            self.__creation_time_label.setText('-')
            self.__start_time_label.setText('-')
            self.__end_time_label.setText('-')
            self.reset_detection_signal.emit()

    def add_detection(self, detection: str):
        scroll_item = QLabel(detection)
        scroll_item.setFixedSize(332, 24)
        scroll_item.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll_item.setFont(self.__scroll_item_font)
        scroll_item.setStyleSheet(self.__scroll_item_ss)
        self.__scroll_container.addItem(scroll_item)

    def reset_detection(self):
        self.__scroll_container.removeAll()
