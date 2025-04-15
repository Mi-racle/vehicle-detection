import imghdr
import os.path
from typing import Optional, Any

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QFontDatabase, QPixmap
from PyQt6.QtWidgets import QWidget, QLabel

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

        self.setFixedSize(450, 1098)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setObjectName('corpusDetailWidget')
        self.setStyleSheet(
            f'{type(self).__name__}#{self.objectName()} {{ background-image: url({settings['background_image']});}}')

        corpus_detail_group = QWidget(self)
        corpus_detail_group.setGeometry(15, 15, 420, 642)

        # corpus_detail_group BEGIN
        self.__corpus_name_background = QWidget(corpus_detail_group)
        self.__corpus_name_background.setGeometry(0, 0, 420, 58)
        self.__corpus_name_background.setObjectName('corpusNameBackground')
        self.__corpus_name_background.setStyleSheet(
            f'QWidget#{self.__corpus_name_background.objectName()} {{ {settings['corpus_name_background_ss']} }}')

        self.__vertical_line = QWidget(corpus_detail_group)
        self.__vertical_line.setGeometry(0, 0, 5, 58)
        self.__vertical_line.setObjectName('verticalLine')
        self.__vertical_line.setStyleSheet(
            f'QWidget#{self.__vertical_line.objectName()} {{ {settings['vertical_line_ss']} }}')

        self.__corpus_name_icon_label = QLabel(corpus_detail_group)
        self.__corpus_name_icon_label.setGeometry(16, 17, 20, 20)
        self.__corpus_name_icon_label.setPixmap(QPixmap(settings['corpus_name_icon']))

        self.__corpus_name_label = QLabel('未选定语料文件', corpus_detail_group)
        self.__corpus_name_label.setGeometry(46, 20, 300, 18)
        self.__corpus_name_label.setFont(font_siyuan_cn_heavy)
        self.__corpus_name_label.setStyleSheet(settings['corpus_name_label_ss'])

        info_title_group = QWidget(corpus_detail_group)
        info_title_group.setGeometry(17, 82, 197, 18)

        # info_title_group BEGIN
        self.__vertical_line_label = QLabel(info_title_group)
        self.__vertical_line_label.setGeometry(0, 0, 5, 18)
        self.__vertical_line_label.setPixmap(QPixmap(settings['vertical_line']))

        self.__info_title_tag_label = QLabel('语料文件信息', info_title_group)
        self.__info_title_tag_label.setGeometry(17, 0, 180, 18)
        self.__info_title_tag_label.setFont(font_siyuan_cn_bold)
        self.__info_title_tag_label.setStyleSheet(settings['info_title_tag_label_ss'])
        # info_title_group END

        model_group = QWidget(corpus_detail_group)
        model_group.setGeometry(17, 108, 386, 78)

        # model_group BEGIN
        self.__model_icon_label = QLabel(model_group)
        self.__model_icon_label.setGeometry(16, 13, 48, 55)
        self.__model_icon_label.setPixmap(QPixmap(settings['model_icon']))

        model_subgroup = QWidget(model_group)
        model_subgroup.setGeometry(80, 18, 306, 43)

        # model_subgroup BEGIN
        self.__model_tag_label = QLabel('模型', model_subgroup)
        self.__model_tag_label.setGeometry(0, 0, 32, 16)
        self.__model_tag_label.setFont(font_siyuan_cn_regular)
        self.__model_tag_label.setStyleSheet(settings['model_tag_label_ss'])

        self.__model_label = QLabel('-', model_subgroup)
        self.__model_label.setGeometry(0, 25, 306, 18)
        self.__model_label.setFont(font_siyuan_cn_regular)
        self.__model_label.setStyleSheet(settings['model_label_ss'])
        # model_subgroup END
        # model_group END

        camera_id_group = QWidget(corpus_detail_group)
        camera_id_group.setGeometry(17, 196, 386, 78)

        # camera_id_group BEGIN
        self.__camera_id_icon_label = QLabel(camera_id_group)
        self.__camera_id_icon_label.setGeometry(16, 14, 48, 48)
        self.__camera_id_icon_label.setPixmap(QPixmap(settings['camera_id_icon']))

        camera_id_subgroup = QWidget(camera_id_group)
        camera_id_subgroup.setGeometry(80, 18, 306, 43)

        # camera_id_subgroup BEGIN
        self.__camera_id_tag_label = QLabel('摄像头', camera_id_subgroup)
        self.__camera_id_tag_label.setGeometry(0, 0, 48, 16)
        self.__camera_id_tag_label.setFont(font_siyuan_cn_regular)
        self.__camera_id_tag_label.setStyleSheet(settings['camera_id_tag_label_ss'])

        self.__camera_id_label = QLabel('-', camera_id_subgroup)
        self.__camera_id_label.setGeometry(0, 25, 306, 18)
        self.__camera_id_label.setFont(font_siyuan_cn_regular)
        self.__camera_id_label.setStyleSheet(settings['camera_id_label_ss'])
        # camera_id_subgroup END
        # camera_id_group END

        video_type_group = QWidget(corpus_detail_group)
        video_type_group.setGeometry(17, 284, 386, 78)

        # video_type_group BEGIN
        self.__video_type_icon_label = QLabel(video_type_group)
        self.__video_type_icon_label.setGeometry(16, 13, 48, 48)
        self.__video_type_icon_label.setPixmap(QPixmap(settings['video_type_icon']))

        video_type_subgroup = QWidget(video_type_group)
        video_type_subgroup.setGeometry(80, 18, 306, 43)

        # video_type_subgroup BEGIN
        self.__video_type_tag_label = QLabel('视频类型', video_type_subgroup)
        self.__video_type_tag_label.setGeometry(0, 0, 64, 16)
        self.__video_type_tag_label.setFont(font_siyuan_cn_regular)
        self.__video_type_tag_label.setStyleSheet(settings['video_type_tag_label_ss'])

        self.__video_type_label = QLabel('-', video_type_subgroup)
        self.__video_type_label.setGeometry(0, 25, 306, 18)
        self.__video_type_label.setFont(font_siyuan_cn_regular)
        self.__video_type_label.setStyleSheet(settings['video_type_label_ss'])
        # video_type_subgroup END
        # video_type_group END

        video_source_group = QWidget(corpus_detail_group)
        video_source_group.setGeometry(17, 372, 386, 78)

        # video_source_group BEGIN
        self.__video_source_icon_label = QLabel(video_source_group)
        self.__video_source_icon_label.setGeometry(5, 12, 68, 51)
        self.__video_source_icon_label.setPixmap(QPixmap(settings['video_source_icon']))

        video_source_subgroup = QWidget(video_source_group)
        video_source_subgroup.setGeometry(80, 18, 306, 44)

        # video_source_subgroup BEGIN
        self.__video_source_tag_label = QLabel('视频源', video_source_subgroup)
        self.__video_source_tag_label.setGeometry(0, 0, 48, 16)
        self.__video_source_tag_label.setFont(font_siyuan_cn_regular)
        self.__video_source_tag_label.setStyleSheet(settings['video_source_tag_label_ss'])

        self.__video_source_label = QLabel('-', video_source_subgroup)
        self.__video_source_label.setGeometry(0, 26, 306, 18)
        self.__video_source_label.setFont(font_siyuan_cn_regular)
        self.__video_source_label.setStyleSheet(settings['video_source_label_ss'])
        # video_source_subgroup END
        # video_source_group END

        time_group = QWidget(corpus_detail_group)
        time_group.setGeometry(17, 490, 386, 72)

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
        self.__dashed_line.setGeometry(0, 71, 386, 1)
        self.__dashed_line.setObjectName('dashedLine')
        self.__dashed_line.setStyleSheet(
            f'QWidget#{self.__dashed_line.objectName()} {{ {settings['dashed_line_ss']} }}')
        # time_group END

        plate_group = QWidget(corpus_detail_group)
        plate_group.setGeometry(17, 586, 386, 16)

        # plate_group BEGIN
        self.__plate_tag_label = QLabel('车牌号', plate_group)
        self.__plate_tag_label.setGeometry(0, 0, 48, 16)
        self.__plate_tag_label.setFont(font_siyuan_cn_regular)
        self.__plate_tag_label.setStyleSheet(settings['plate_tag_label_ss'])

        self.__plate_label = QLabel('-', plate_group)
        self.__plate_label.setGeometry(48, 0, 338, 16)
        self.__plate_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.__plate_label.setFont(font_siyuan_cn_regular)
        self.__plate_label.setStyleSheet(settings['plate_label_ss'])
        # plate_group END

        position_group = QWidget(corpus_detail_group)
        position_group.setGeometry(17, 626, 386, 16)

        # position_group BEGIN
        self.__position_tag_label = QLabel('位置', position_group)
        self.__position_tag_label.setGeometry(0, 0, 48, 16)
        self.__position_tag_label.setFont(font_siyuan_cn_regular)
        self.__position_tag_label.setStyleSheet(settings['position_tag_label_ss'])

        self.__position_label = QLabel('-', position_group)
        self.__position_label.setGeometry(48, 0, 338, 16)
        self.__position_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.__position_label.setFont(font_siyuan_cn_regular)
        self.__position_label.setStyleSheet(settings['position_label_ss'])
        # position_group END
        # corpus_detail_group END

        display_title_group = QWidget(self)
        display_title_group.setGeometry(32, 697, 197, 18)

        # display_title_group BEGIN
        self.__vertical_line2_label = QLabel(display_title_group)
        self.__vertical_line2_label.setGeometry(0, 0, 5, 18)
        self.__vertical_line2_label.setPixmap(QPixmap(settings['vertical_line2']))

        self.__display_title_tag_label = QLabel('语料文件展示', display_title_group)
        self.__display_title_tag_label.setGeometry(17, 0, 180, 18)
        self.__display_title_tag_label.setFont(font_siyuan_cn_bold)
        self.__display_title_tag_label.setStyleSheet(settings['display_title_tag_label_ss'])
        # display_title_group END

        display_group = QWidget(self)
        display_group.setGeometry(16, 735, 418, 323)

        # display_group BEGIN
        self.__display_background = QLabel(display_group)
        self.__display_background.setGeometry(0, 0, 418, 323)
        self.__display_background.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.__display_background.setObjectName('displayBackground')
        self.__display_background.setStyleSheet(
            f'''
            QLabel#{self.__display_background.objectName()} {{ 
            background-image: url({settings['display_background_image']});
            }}
            '''
        )

        self.__display_label = ImageLabel(parent=display_group)
        self.__display_label.setGeometry(0, 0, 418, 323)

        loading_group = QWidget(display_group)
        loading_group.setGeometry(119, 142, 181, 40)

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

        self.__corpus_dir = corpus_dir

    def set_corpus(self, corpus_entry: dict | None = None):
        if corpus_entry:
            self.__corpus_name_label.setText(corpus_entry['dest'])
            self.__model_label.setText(f'{corpus_entry['model_name']}v{corpus_entry['model_version']}')
            self.__camera_id_label.setText(corpus_entry['camera_id'])
            self.__video_type_label.setText('实时视频流' if int(corpus_entry['video_type']) == 1 else '视频文件')
            self.__video_source_label.setText(corpus_entry['source'])
            self.__start_time_label.setText(str(corpus_entry['start_time']))  # TODO may need concat date and time
            self.__end_time_label.setText(str(corpus_entry['end_time']))  # TODO may need concat date and time
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
            self.__camera_id_label.setText('-')
            self.__video_type_label.setText('-')
            self.__video_source_label.setText('-')
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
