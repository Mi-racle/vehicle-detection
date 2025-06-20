from typing import Optional, Callable

from PyQt6 import sip
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QFontDatabase, QPixmap, QResizeEvent
from PyQt6.QtWidgets import QWidget, QLabel

from ui.ui_utils import ScrollContainer, ScrollAreaWithShift
from utils import LimitedDict


class CorpusListWidget(QWidget):
    send_select_signal = pyqtSignal(dict)
    send_unselect_signal = pyqtSignal(object)

    class ScrollItemWidget(QWidget):
        def __init__(
                self,
                settings: dict,
                comm: Callable[[str, bool], None],
                corpus_name='-',
                position='-',
                model='-',
                start_time='-',
                end_time='-',
                parent: Optional[QWidget] = None
        ):
            super().__init__(parent)

            font_families = QFontDatabase.families()
            self.__font_siyuan_cn_heavy = (
                QFont('Source Han Sans CN Heavy')) if 'Source Han Sans CN Heavy' in font_families else QFont(
                    QFontDatabase.applicationFontFamilies(
                        QFontDatabase.addApplicationFont(settings['font_siyuan_cn_heavy']))[0])
            self.__font_siyuan_cn_medium = (
                QFont('Source Han Sans CN Medium')) if 'Source Han Sans CN Medium' in font_families else QFont(
                    QFontDatabase.applicationFontFamilies(
                        QFontDatabase.addApplicationFont(settings['font_siyuan_cn_medium']))[0])
            self.__font_siyuan_cn_regular = (
                QFont('Source Han Sans CN')) if 'Source Han Sans CN' in font_families else QFont(
                    QFontDatabase.applicationFontFamilies(
                        QFontDatabase.addApplicationFont(settings['font_siyuan_cn_regular']))[0])

            self.setFixedSize(896, 84)
            self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
            self.setObjectName(corpus_name)

            position_label_width_bias = int(settings['position_label_width_bias'])
            model_label_width_bias = int(settings['model_label_width_bias'])

            self.__vertical_line_label = QLabel(self)
            self.__vertical_line_label.setGeometry(0, 0, 5, 84)
            self.__vertical_line_label.setPixmap(QPixmap(settings['vertical_line']))
            self.__vertical_line_label.setVisible(False)

            corpus_info_group = QWidget(self)
            corpus_info_group.setGeometry(20, 18, 876, 46)

            # corpus_info_group BEGIN
            self.__unselected_icon_pixmap = QPixmap(settings['scroll_item_icon'])
            self.__selected_icon_pixmap = QPixmap(settings['scroll_item_selected_icon'])
            self.__icon_label = QLabel(corpus_info_group)
            self.__icon_label.setGeometry(0, 0, 20, 20)
            self.__icon_label.setPixmap(self.__unselected_icon_pixmap)

            self.__corpus_name_label = QLabel(corpus_name, corpus_info_group)
            self.__corpus_name_label.setGeometry(30, 2, 300, 18)
            self.__corpus_name_label.setFont(self.__font_siyuan_cn_medium)
            self.__corpus_name_label.setStyleSheet(settings['corpus_name_label_ss'])

            position_group = QWidget(corpus_info_group)
            position_group.setGeometry(0, 30, 90 + position_label_width_bias, 16)

            # position_group BEGIN
            self.__position_tag_label = QLabel('位置｜', position_group)
            self.__position_tag_label.setGeometry(0, 0, 48, 16)
            self.__position_tag_label.setFont(self.__font_siyuan_cn_regular)
            self.__position_tag_label.setStyleSheet(settings['position_tag_label_ss'])

            self.__position_label = QLabel(position, position_group)
            self.__position_label.setGeometry(48, 0, 42 + position_label_width_bias, 16)
            self.__position_label.setFont(self.__font_siyuan_cn_regular)
            self.__position_label.setStyleSheet(settings['position_label_ss'])
            # position_group END

            model_group = QWidget(corpus_info_group)
            model_group.setGeometry(114 + position_label_width_bias, 30, 144 + model_label_width_bias, 16)

            # model_group BEGIN
            self.__model_tag_label = QLabel('模型｜', model_group)
            self.__model_tag_label.setGeometry(0, 0, 48, 16)
            self.__model_tag_label.setFont(self.__font_siyuan_cn_regular)
            self.__model_tag_label.setStyleSheet(settings['model_tag_label_ss'])

            self.__model_label = QLabel(model, model_group)
            self.__model_label.setGeometry(48, 0, 96 + model_label_width_bias, 16)
            self.__model_label.setFont(self.__font_siyuan_cn_regular)
            self.__model_label.setStyleSheet(settings['model_label_ss'])
            # model_group END

            start_time_group = QWidget(corpus_info_group)
            start_time_group.setGeometry(282 + position_label_width_bias + model_label_width_bias, 30, 243, 16)

            # start_time_group BEGIN
            self.__start_time_tag_label = QLabel('开始时间｜', start_time_group)
            self.__start_time_tag_label.setGeometry(0, 0, 80, 16)
            self.__start_time_tag_label.setFont(self.__font_siyuan_cn_regular)
            self.__start_time_tag_label.setStyleSheet(settings['start_time_tag_label_ss'])

            self.__start_time_label = QLabel(start_time, start_time_group)
            self.__start_time_label.setGeometry(80, 0, 163, 16)
            self.__start_time_label.setFont(self.__font_siyuan_cn_regular)
            self.__start_time_label.setStyleSheet(settings['start_time_label_ss'])
            # start_time_group END

            end_time_group = QWidget(corpus_info_group)
            end_time_group.setGeometry(533 + position_label_width_bias + model_label_width_bias, 30, 243, 16)

            # end_time_group BEGIN
            self.__end_time_tag_label = QLabel('结束时间｜', end_time_group)
            self.__end_time_tag_label.setGeometry(0, 0, 80, 16)
            self.__end_time_tag_label.setFont(self.__font_siyuan_cn_regular)
            self.__end_time_tag_label.setStyleSheet(settings['end_time_tag_label_ss'])

            self.__end_time_label = QLabel(end_time, end_time_group)
            self.__end_time_label.setGeometry(80, 0, 163, 16)
            self.__end_time_label.setFont(self.__font_siyuan_cn_regular)
            self.__end_time_label.setStyleSheet(settings['end_time_label_ss'])
            # end_time_group END
            # corpus_info_group END

            self.__dashed_line = QWidget(self)
            self.__dashed_line.setGeometry(0, 83, 896, 1)
            self.__dashed_line.setObjectName('dashedLine')
            self.__dashed_line.setStyleSheet(
                f'QWidget#{self.__dashed_line.objectName()} {{ {settings['dashed_line_ss']} }}')

            self.__comm = comm
            self.__settings = settings
            self.__selected = False

        def mousePressEvent(self, event):
            if event.button() != Qt.MouseButton.LeftButton:
                return

            self.select(not self.__selected)  # flip self.__selected

        def enterEvent(self, event):
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.__change_style(True)
            super().enterEvent(event)

        def leaveEvent(self, event):
            self.setCursor(Qt.CursorShape.ArrowCursor)
            if not self.__selected:
                self.__change_style(False)
            super().leaveEvent(event)

        def select(self, selected: bool):
            self.__selected = selected
            self.__change_style(selected)
            self.__comm(self.__corpus_name_label.text(), selected)

        def __change_style(self, selected: bool):
            if selected:
                self.setStyleSheet(
                    f'{type(self).__name__}#{self.objectName()} {{ {self.__settings['scroll_item_selected_ss']} }}')
                self.__vertical_line_label.setVisible(True)
                self.__icon_label.setPixmap(self.__selected_icon_pixmap)
                self.__corpus_name_label.setFont(self.__font_siyuan_cn_heavy)
                self.__corpus_name_label.setStyleSheet(self.__settings['corpus_name_selected_label_ss'])
                self.__position_tag_label.setStyleSheet(self.__settings['position_tag_selected_label_ss'])
                self.__position_label.setStyleSheet(self.__settings['position_selected_label_ss'])
                self.__model_tag_label.setStyleSheet(self.__settings['model_tag_selected_label_ss'])
                self.__model_label.setStyleSheet(self.__settings['model_selected_label_ss'])
                self.__start_time_tag_label.setStyleSheet(self.__settings['start_time_selected_tag_label_ss'])
                self.__start_time_label.setStyleSheet(self.__settings['start_time_selected_label_ss'])
                self.__end_time_tag_label.setStyleSheet(self.__settings['end_time_selected_tag_label_ss'])
                self.__end_time_label.setStyleSheet(self.__settings['end_time_selected_label_ss'])
                self.__dashed_line.setVisible(False)

            else:
                self.setStyleSheet(f'{type(self).__name__}#{self.objectName()} {{ }}')
                self.__vertical_line_label.setVisible(False)
                self.__icon_label.setPixmap(self.__unselected_icon_pixmap)
                self.__corpus_name_label.setFont(self.__font_siyuan_cn_medium)
                self.__corpus_name_label.setStyleSheet(self.__settings['corpus_name_label_ss'])
                self.__position_tag_label.setStyleSheet(self.__settings['position_tag_label_ss'])
                self.__position_label.setStyleSheet(self.__settings['position_label_ss'])
                self.__model_tag_label.setStyleSheet(self.__settings['model_tag_label_ss'])
                self.__model_label.setStyleSheet(self.__settings['model_label_ss'])
                self.__start_time_tag_label.setStyleSheet(self.__settings['start_time_tag_label_ss'])
                self.__start_time_label.setStyleSheet(self.__settings['start_time_label_ss'])
                self.__end_time_tag_label.setStyleSheet(self.__settings['end_time_tag_label_ss'])
                self.__end_time_label.setStyleSheet(self.__settings['end_time_label_ss'])
                self.__dashed_line.setVisible(True)

    def __init__(self, settings: dict, parent: Optional[QWidget] = None):
        super().__init__(parent)

        font_families = QFontDatabase.families()
        font_siyuan_cn_bold = (
            QFont('Source Han Sans CN Bold')) if 'Source Han Sans CN Bold' in font_families else QFont(
                QFontDatabase.applicationFontFamilies(
                    QFontDatabase.addApplicationFont(settings['font_siyuan_cn_bold']))[0])
        font_siyuan_cn_regular = (
            QFont('Source Han Sans CN')) if 'Source Han Sans CN' in font_families else QFont(
                QFontDatabase.applicationFontFamilies(
                    QFontDatabase.addApplicationFont(settings['font_siyuan_cn_regular']))[0])

        self.__background = QLabel(self)
        self.__background.setGeometry(0, 0, 960, 349)
        self.__background.setPixmap(QPixmap(settings['background_image']))
        self.__background.setScaledContents(True)

        title_group = QWidget(self)
        title_group.setGeometry(16, 16, 928, 44)

        # title_group BEGIN
        self.__title_icon_label = QLabel(title_group)
        self.__title_icon_label.setGeometry(8, 4, 22, 22)
        self.__title_icon_label.setPixmap(QPixmap(settings['title_icon']))

        self.__title_tag_label = QLabel('已生成的语料文件', title_group)
        self.__title_tag_label.setGeometry(38, 4, 192, 22)
        self.__title_tag_label.setFont(font_siyuan_cn_bold)
        self.__title_tag_label.setStyleSheet(settings['title_tag_label_ss'])

        self.__title_line_label = QLabel(title_group)
        self.__title_line_label.setGeometry(8, 41, 912, 3)
        self.__title_line_label.setPixmap(QPixmap(settings['title_line']))
        self.__title_line_label.setScaledContents(True)
        # title_group END

        self.__tip_tag_label = QLabel('暂无语料文件', self)
        self.__tip_tag_label.setGeometry(32, 190, 896, 18)
        self.__tip_tag_label.setFont(font_siyuan_cn_regular)
        self.__tip_tag_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.__tip_tag_label.setStyleSheet(settings['tip_tag_label_ss'])

        self.__scroll_container = ScrollContainer(maxlen=settings['scroll_maxlen'], reverse=True)
        self.__scroll_container.setFixedSize(0, 0)

        self.__scroll_area = ScrollAreaWithShift(self)
        self.__scroll_area.setGeometry(32, 74, 896 + 24, 84 * 3)  # 24 = 14+8+1*2 = gap + bar_width + bar_border * 2
        self.__scroll_area.setWidget(self.__scroll_container)
        self.__scroll_area.setFrameShape(ScrollAreaWithShift.Shape.NoFrame)
        self.__scroll_area.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.__scroll_area.setStyleSheet(f'''
            {type(self.__scroll_area).__name__} {{ background: transparent;}}
            {settings['scroll_bar_ss']}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; background: none;}}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none;}}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ height: 0px; background: none;}}
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: none;}}
        ''')

        self.__settings = settings
        self.__corpus_entries = LimitedDict(maxlen=settings['scroll_maxlen'])
        self.__selected_item: CorpusListWidget.ScrollItemWidget | None = None

        # self.__scroll_test(5)

    def resizeEvent(self, event: QResizeEvent):
        old_size = event.oldSize()
        if old_size.width() == -1 or old_size.height() == -1:
            super().resizeEvent(event)
            return

        new_size = event.size()
        width_diff = old_size.width() - new_size.width()

        self.__background.resize(new_size)
        self.__title_line_label.setFixedWidth(self.__title_line_label.width() - width_diff)
        self.__tip_tag_label.setFixedWidth(self.__tip_tag_label.width() - width_diff)
        self.__scroll_area.setFixedWidth(self.__scroll_area.width() - width_diff)

        super().resizeEvent(event)

    def add_corpus(self, corpus_entry: dict, camera_position: str):
        item = CorpusListWidget.ScrollItemWidget(
            self.__settings['scroll_item'],
            self.__comm,
            str(corpus_entry['dest']),
            str(camera_position),
            str(corpus_entry['model_name']),
            str(corpus_entry['start_time']),
            str(corpus_entry['end_time'])
        )
        item.setFixedSize(896, 84)
        self.__scroll_container.addItem(item)
        self.__corpus_entries[corpus_entry['dest']] = corpus_entry
        self.__tip_tag_label.setVisible(False)

    def __send_corpus(self, corpus_entry: dict | None = None):
        """
        Send corpus to corpus_detail. None | [] for reset.
        :param corpus_entry: corpus info as list or None
        """
        emit_func = self.send_select_signal.emit if corpus_entry else self.send_unselect_signal.emit
        emit_func(corpus_entry)

    def __comm(self, object_name: str, selected: bool):
        """
        Callback by scroll item
        :param object_name: QWidget name === corpus name
        :param selected: select or unselect
        """
        corpus_entry: dict | None = None

        if selected:
            if self.__selected_item and not sip.isdeleted(self.__selected_item):
                self.__selected_item.select(False)

            self.__selected_item: CorpusListWidget.ScrollItemWidget = self.__scroll_container.getItem(object_name)
            corpus_entry = self.__corpus_entries[object_name]

        else:
            self.__selected_item = None

        self.__send_corpus(corpus_entry)

    def __scroll_test(self, item_num: int):
        """ Unit test """
        for i in range(item_num):
            self.add_corpus(
                {
                    'model_name': 'parking',
                    'model_version': '1.0.0',
                    'camera_type': 1,
                    'camera_id': '1z2x3c4v5b6n7m8j9k0l',
                    'video_type': 1,
                    'source': 'D:/xxs-signs/vehicle-detection/resources/hangzhou.mp4',
                    'dest': f'ba76cqgj9112862972eal4{i}.mp4',
                    'start_time': '2025-12-31 23:23:23',
                    'end_time': '2025-12-31 23:23:59',
                    'plate_no': None,
                    'locations': '杭州市闻涛路口'
                },
                '杭州市闻涛路口'
            )
