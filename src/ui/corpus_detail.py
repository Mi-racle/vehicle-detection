import imghdr
import os.path
from typing import Optional, Any

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QFontDatabase, QPixmap, QResizeEvent
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout

from ui.ui_utils import ImageLabel


class CorpusDetailWidget(QWidget):
    def __init__(self, settings: dict, corpus_dir='', parent: Optional[QWidget] = None):
        super().__init__(parent)

        font_siyuan_cn_heavy = QFont(
            QFontDatabase.applicationFontFamilies(
                QFontDatabase.addApplicationFont(settings['font_siyuan_cn_heavy']))[0])
        font_siyuan_cn_bold = QFont(
            QFontDatabase.applicationFontFamilies(
                QFontDatabase.addApplicationFont(settings['font_siyuan_cn_bold']))[0])
        font_siyuan_cn_regular = QFont(
            QFontDatabase.applicationFontFamilies(
                QFontDatabase.addApplicationFont(settings['font_siyuan_cn_regular']))[0])

        self.__background = QLabel(self)
        self.__background.setGeometry(0, 0, 450, 1014)
        self.__background.setPixmap(QPixmap(settings['background_image']))
        self.__background.setScaledContents(True)

        corpus_name_group = QWidget()
        corpus_name_group.setFixedSize(420, 58)

        # corpus_name_group BEGIN
        self.__corpus_name_background = QWidget(corpus_name_group)
        self.__corpus_name_background.setGeometry(0, 0, 420, 58)
        self.__corpus_name_background.setObjectName('corpusNameBackground')
        self.__corpus_name_background.setStyleSheet(
            f'QWidget#{self.__corpus_name_background.objectName()} {{ {settings['corpus_name_background_ss']} }}')

        self.__vertical_line = QWidget(corpus_name_group)
        self.__vertical_line.setGeometry(0, 0, 5, 58)
        self.__vertical_line.setObjectName('verticalLine')
        self.__vertical_line.setStyleSheet(
            f'QWidget#{self.__vertical_line.objectName()} {{ {settings['vertical_line_ss']} }}')

        self.__corpus_name_icon_label = QLabel(corpus_name_group)
        self.__corpus_name_icon_label.setGeometry(16, 17, 20, 20)
        self.__corpus_name_icon_label.setPixmap(QPixmap(settings['corpus_name_icon']))

        self.__corpus_name_label = QLabel('未选定语料文件', corpus_name_group)
        self.__corpus_name_label.setGeometry(46, 20, 300, 18)
        self.__corpus_name_label.setFont(font_siyuan_cn_heavy)
        self.__corpus_name_label.setStyleSheet(settings['corpus_name_label_ss'])
        # corpus_name_group END

        info_title_group = QWidget()
        info_title_group.setFixedSize(420, 18)

        # info_title_group BEGIN
        self.__vertical_line_label = QLabel(info_title_group)
        self.__vertical_line_label.setGeometry(17, 0, 5, 18)
        self.__vertical_line_label.setPixmap(QPixmap(settings['vertical_line']))

        self.__info_title_tag_label = QLabel('语料文件信息', info_title_group)
        self.__info_title_tag_label.setGeometry(34, 0, 180, 18)
        self.__info_title_tag_label.setFont(font_siyuan_cn_bold)
        self.__info_title_tag_label.setStyleSheet(settings['info_title_tag_label_ss'])
        # info_title_group END

        model_group = QWidget()
        model_group.setFixedSize(420, 48)

        # model_group BEGIN
        self.__model_icon_label = QLabel(model_group)
        self.__model_icon_label.setGeometry(33, -5, 48, 55)
        self.__model_icon_label.setPixmap(QPixmap(settings['model_icon']))

        model_subgroup = QWidget(model_group)
        model_subgroup.setGeometry(97, 5, 306, 43)

        # model_subgroup BEGIN
        self.__model_tag_label = QLabel('模型', model_subgroup)
        self.__model_tag_label.setGeometry(0, 0, 32, 16)
        self.__model_tag_label.setFont(font_siyuan_cn_regular)
        self.__model_tag_label.setStyleSheet(settings['model_tag_label_ss'])

        self.__model_label = QLabel('-', model_subgroup)
        self.__model_label.setGeometry(0, 25, 306, 18)
        self.__model_label.setToolTip(self.__model_label.text())
        self.__model_label.setFont(font_siyuan_cn_regular)
        self.__model_label.setStyleSheet(settings['model_label_ss'])
        # model_subgroup END
        # model_group END

        camera_id_group = QWidget()
        camera_id_group.setFixedSize(420, 48)

        # camera_id_group BEGIN
        self.__camera_id_icon_label = QLabel(camera_id_group)
        self.__camera_id_icon_label.setGeometry(33, 0, 48, 48)
        self.__camera_id_icon_label.setPixmap(QPixmap(settings['camera_id_icon']))

        camera_id_subgroup = QWidget(camera_id_group)
        camera_id_subgroup.setGeometry(97, 5, 306, 43)

        # camera_id_subgroup BEGIN
        self.__camera_id_tag_label = QLabel('摄像头', camera_id_subgroup)
        self.__camera_id_tag_label.setGeometry(0, 0, 48, 16)
        self.__camera_id_tag_label.setFont(font_siyuan_cn_regular)
        self.__camera_id_tag_label.setStyleSheet(settings['camera_id_tag_label_ss'])

        self.__camera_id_label = QLabel('-', camera_id_subgroup)
        self.__camera_id_label.setGeometry(0, 25, 306, 18)
        self.__camera_id_label.setToolTip(self.__camera_id_label.text())
        self.__camera_id_label.setFont(font_siyuan_cn_regular)
        self.__camera_id_label.setStyleSheet(settings['camera_id_label_ss'])
        # camera_id_subgroup END
        # camera_id_group END

        video_type_group = QWidget()
        video_type_group.setFixedSize(420, 48)

        # video_type_group BEGIN
        self.__video_type_icon_label = QLabel(video_type_group)
        self.__video_type_icon_label.setGeometry(33, 0, 48, 48)
        self.__video_type_icon_label.setPixmap(QPixmap(settings['video_type_icon']))

        video_type_subgroup = QWidget(video_type_group)
        video_type_subgroup.setGeometry(97, 5, 306, 43)

        # video_type_subgroup BEGIN
        self.__video_type_tag_label = QLabel('视频类型', video_type_subgroup)
        self.__video_type_tag_label.setGeometry(0, 0, 64, 16)
        self.__video_type_tag_label.setFont(font_siyuan_cn_regular)
        self.__video_type_tag_label.setStyleSheet(settings['video_type_tag_label_ss'])

        self.__video_type_label = QLabel('-', video_type_subgroup)
        self.__video_type_label.setGeometry(0, 25, 306, 18)
        self.__video_type_label.setToolTip(self.__video_type_label.text())
        self.__video_type_label.setFont(font_siyuan_cn_regular)
        self.__video_type_label.setStyleSheet(settings['video_type_label_ss'])
        # video_type_subgroup END
        # video_type_group END

        video_source_group = QWidget()
        video_source_group.setFixedSize(420, 48)

        # video_source_group BEGIN
        self.__video_source_icon_label = QLabel(video_source_group)
        self.__video_source_icon_label.setGeometry(33, 0, 48, 48)
        self.__video_source_icon_label.setPixmap(QPixmap(settings['video_source_icon']))

        video_source_subgroup = QWidget(video_source_group)
        video_source_subgroup.setGeometry(97, 5, 306, 43)

        # video_source_subgroup BEGIN
        self.__video_source_tag_label = QLabel('视频源', video_source_subgroup)
        self.__video_source_tag_label.setGeometry(0, 0, 48, 16)
        self.__video_source_tag_label.setFont(font_siyuan_cn_regular)
        self.__video_source_tag_label.setStyleSheet(settings['video_source_tag_label_ss'])

        self.__video_source_label = QLabel('-', video_source_subgroup)
        self.__video_source_label.setGeometry(0, 25, 306, 18)
        self.__video_source_label.setToolTip(self.__video_source_label.text())
        self.__video_source_label.setFont(font_siyuan_cn_regular)
        self.__video_source_label.setStyleSheet(settings['video_source_label_ss'])
        # video_source_subgroup END
        # video_source_group END

        time_group = QWidget()
        time_group.setFixedSize(420, 47)

        # time_group BEGIN
        start_time_group = QWidget(time_group)
        start_time_group.setGeometry(17, 0, 170, 47)

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
        end_time_group.setGeometry(220, 0, 170, 47)

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
        dashed_line_group.setFixedSize(420, 1)

        # dashed_line_group BEGIN
        self.__dashed_line = QWidget(dashed_line_group)
        self.__dashed_line.setGeometry(17, 0, 386, 1)
        self.__dashed_line.setObjectName('dashedLine')
        self.__dashed_line.setStyleSheet(
            f'QWidget#{self.__dashed_line.objectName()} {{ {settings['dashed_line_ss']} }}')
        # dashed_line_group END

        plate_group = QWidget()
        plate_group.setFixedSize(420, 16)

        # plate_group BEGIN
        self.__plate_tag_label = QLabel('车牌号', plate_group)
        self.__plate_tag_label.setGeometry(17, 0, 48, 16)
        self.__plate_tag_label.setFont(font_siyuan_cn_regular)
        self.__plate_tag_label.setStyleSheet(settings['plate_tag_label_ss'])

        self.__plate_label = QLabel('-', plate_group)
        self.__plate_label.setGeometry(65, 0, 338, 16)
        self.__plate_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.__plate_label.setFont(font_siyuan_cn_regular)
        self.__plate_label.setStyleSheet(settings['plate_label_ss'])
        # plate_group END

        position_group = QWidget()
        position_group.setFixedSize(420, 16)

        # position_group BEGIN
        self.__position_tag_label = QLabel('位置', position_group)
        self.__position_tag_label.setGeometry(17, 0, 48, 16)
        self.__position_tag_label.setFont(font_siyuan_cn_regular)
        self.__position_tag_label.setStyleSheet(settings['position_tag_label_ss'])

        self.__position_label = QLabel('-', position_group)
        self.__position_label.setGeometry(65, 0, 338, 16)
        self.__position_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.__position_label.setFont(font_siyuan_cn_regular)
        self.__position_label.setStyleSheet(settings['position_label_ss'])
        # position_group END

        display_title_group = QWidget()
        display_title_group.setFixedSize(420, 18)

        # display_title_group BEGIN
        self.__vertical_line2_label = QLabel(display_title_group)
        self.__vertical_line2_label.setGeometry(17, 0, 5, 18)
        self.__vertical_line2_label.setPixmap(QPixmap(settings['vertical_line2']))

        self.__display_title_tag_label = QLabel('语料文件展示', display_title_group)
        self.__display_title_tag_label.setGeometry(34, 0, 180, 18)
        self.__display_title_tag_label.setFont(font_siyuan_cn_bold)
        self.__display_title_tag_label.setStyleSheet(settings['display_title_tag_label_ss'])
        # display_title_group END

        display_group = QWidget()
        display_group.setFixedSize(420, 257)

        self.__display_background = QLabel(display_group)
        self.__display_background.setGeometry(0, 0, 420, 257)
        self.__display_background.setPixmap(QPixmap(settings['display_background_image']))
        self.__display_background.setScaledContents(True)

        self.__display_label = ImageLabel(parent=display_group)
        self.__display_label.setGeometry(0, 0, 420, 257)

        loading_group = QWidget(display_group)
        loading_group.setGeometry(119, 109, 181, 40)

        # loading_group BEGIN
        self.__loading_icon_label = QLabel(loading_group)
        self.__loading_icon_label.setGeometry(0, 0, 40, 40)
        self.__loading_icon_label.setPixmap(QPixmap(settings['loading_icon']))

        self.__loading_tag_label = QLabel('未选定语料文件', loading_group)
        self.__loading_tag_label.setGeometry(48, 11, 133, 18)
        self.__loading_tag_label.setFont(font_siyuan_cn_regular)
        self.__loading_tag_label.setStyleSheet(settings['loading_tag_label_ss'])
        # loading_group END
        # display_group END

        vertical_layout = QVBoxLayout(self)
        vertical_layout.setSpacing(0)
        vertical_layout.setContentsMargins(15, 15, 15, 22)

        vertical_layout.addWidget(corpus_name_group)
        vertical_layout.addStretch(24)
        vertical_layout.addWidget(info_title_group)
        vertical_layout.addStretch(21)
        vertical_layout.addWidget(model_group)
        vertical_layout.addStretch(40)
        vertical_layout.addWidget(camera_id_group)
        vertical_layout.addStretch(40)
        vertical_layout.addWidget(video_type_group)
        vertical_layout.addStretch(40)
        vertical_layout.addWidget(video_source_group)
        vertical_layout.addStretch(56)
        vertical_layout.addWidget(time_group)
        vertical_layout.addStretch(24)
        vertical_layout.addWidget(dashed_line_group)
        vertical_layout.addStretch(24)
        vertical_layout.addWidget(plate_group)
        vertical_layout.addStretch(24)
        vertical_layout.addWidget(position_group)
        vertical_layout.addStretch(40)
        vertical_layout.addWidget(display_title_group)
        vertical_layout.addStretch(20)
        vertical_layout.addWidget(display_group)

        self.__corpus_dir = corpus_dir

    def resizeEvent(self, event: QResizeEvent):
        if event.oldSize().width() != -1 and event.oldSize().height() != -1:
            self.__background.resize(event.size())

        super().resizeEvent(event)

    def set_corpus(self, corpus_entry: dict | None = None):
        if corpus_entry:
            self.__corpus_name_label.setText(corpus_entry['dest'])
            self.__model_label.setText(f'{corpus_entry['model_name']}v{corpus_entry['model_version']}')
            self.__model_label.setToolTip(self.__model_label.text())
            self.__camera_id_label.setText(corpus_entry['camera_id'])
            self.__camera_id_label.setToolTip(self.__camera_id_label.text())
            self.__video_type_label.setText('实时视频流' if int(corpus_entry['video_type']) == 1 else '视频文件')
            self.__video_type_label.setToolTip(self.__video_type_label.text())
            self.__video_source_label.setText(corpus_entry['source'])
            self.__video_source_label.setToolTip(self.__video_source_label.text())
            self.__start_time_label.setText(str(corpus_entry['start_time']))
            self.__end_time_label.setText(str(corpus_entry['end_time']))
            self.__plate_label.setText(corpus_entry['plate_no'] if corpus_entry['plate_no'] else '无')
            self.__position_label.setText(corpus_entry['locations'])
            self.__loading_icon_label.setVisible(False)
            self.__loading_tag_label.setVisible(False)
            umat = self.__capture_image(corpus_entry['dest'])
            if umat is not None:
                self.__display_label.setUmat(umat)
            else:
                self.__loading_tag_label.setText('未找到语料文件')
                self.__display_label.reset()

        else:  # reset
            self.__corpus_name_label.setText('未选定语料文件')
            self.__model_label.setText('-')
            self.__model_label.setToolTip(self.__model_label.text())
            self.__camera_id_label.setText('-')
            self.__camera_id_label.setToolTip(self.__camera_id_label.text())
            self.__video_type_label.setText('-')
            self.__video_type_label.setToolTip(self.__video_type_label.text())
            self.__video_source_label.setText('-')
            self.__video_source_label.setToolTip(self.__video_source_label.text())
            self.__start_time_label.setText('-')
            self.__end_time_label.setText('-')
            self.__plate_label.setText('-')
            self.__position_label.setText('-')
            self.__loading_icon_label.setVisible(True)
            self.__loading_tag_label.setVisible(True)
            self.__loading_tag_label.setText('未选定语料文件')
            self.__display_label.reset()

    def __capture_image(self, dest: str) -> cv2.Mat | np.ndarray[Any, np.dtype] | np.ndarray | None:
        corpus_path = f'{self.__corpus_dir}/{dest}'

        if not os.path.exists(corpus_path):
            return None

        try:
            umat = None

            if imghdr.what(corpus_path):
                umat = cv2.imread(corpus_path)

            else:
                cap = cv2.VideoCapture(corpus_path)
                ret, frame = cap.read()

                if ret:
                    umat = frame

                cap.release()

            return umat

        except Exception as e:
            print(e)
