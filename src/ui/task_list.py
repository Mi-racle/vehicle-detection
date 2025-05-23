from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QFontDatabase, QPixmap
from PyQt6.QtWidgets import QWidget, QLabel

from ui.ui_utils import ScrollContainer, ScrollAreaWithShift


class TaskListWidget(QWidget):

    class ScrollItemWidget(QWidget):
        def __init__(self, settings: dict, task_name='-', start_time='-', parent: Optional[QWidget] = None):
            super().__init__(parent)

            font_families = QFontDatabase.families()
            font_siyuan_cn_medium = (
                QFont('Source Han Sans CN Medium')) if 'Source Han Sans CN Medium' in font_families else QFont(
                    QFontDatabase.applicationFontFamilies(
                        QFontDatabase.addApplicationFont(settings['font_siyuan_cn_medium']))[0])
            font_siyuan_cn_regular = (
                QFont('Source Han Sans CN')) if 'Source Han Sans CN' in font_families else QFont(
                    QFontDatabase.applicationFontFamilies(
                        QFontDatabase.addApplicationFont(settings['font_siyuan_cn_regular']))[0])

            self.setFixedSize(386, 84)
            self.setObjectName(task_name)

            self.__icon_label = QLabel(self)
            self.__icon_label.setGeometry(0, 21, 20, 20)
            self.__icon_label.setPixmap(QPixmap(settings['scroll_item_icon']))

            task_info_group = QWidget(self)
            task_info_group.setGeometry(30, 20, 260, 44)

            # task_info_group BEGIN
            self.__task_name_label = QLabel(task_name, task_info_group)
            self.__task_name_label.setGeometry(0, 0, 260, 18)
            self.__task_name_label.setFont(font_siyuan_cn_medium)
            self.__task_name_label.setStyleSheet(settings['task_name_label_ss'])

            start_time_group = QWidget(task_info_group)
            start_time_group.setGeometry(0, 28, 260, 16)

            # start_time_group BEGIN
            self.__start_time_tag_label = QLabel('开始时间｜', start_time_group)
            self.__start_time_tag_label.setGeometry(0, 0, 80, 16)
            self.__start_time_tag_label.setFont(font_siyuan_cn_regular)
            self.__start_time_tag_label.setStyleSheet(settings['start_time_tag_label_ss'])

            self.__start_time_label = QLabel(start_time, start_time_group)
            self.__start_time_label.setGeometry(80, 0, 180, 16)
            self.__start_time_label.setFont(font_siyuan_cn_regular)
            self.__start_time_label.setStyleSheet(settings['start_time_label_ss'])
            # start_time_group END
            # task_info_group END

            self.__dashed_line = QWidget(self)
            self.__dashed_line.setGeometry(0, 83, 386, 1)
            self.__dashed_line.setObjectName('dashedLine')
            self.__dashed_line.setStyleSheet(
                f'QWidget#{self.__dashed_line.objectName()} {{ {settings['dashed_line_ss']} }}')

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
        self.__background.setGeometry(0, 0, 450, 349)
        self.__background.setPixmap(QPixmap(settings['background_image']))
        self.__background.setScaledContents(True)

        title_group = QWidget(self)
        title_group.setGeometry(16, 16, 418, 44)

        # title_group BEGIN
        self.__title_icon_label = QLabel(title_group)
        self.__title_icon_label.setGeometry(8, 4, 22, 22)
        self.__title_icon_label.setPixmap(QPixmap(settings['title_icon']))

        self.__title_tag_label = QLabel('待执行任务', title_group)
        self.__title_tag_label.setGeometry(38, 4, 120, 22)
        self.__title_tag_label.setFont(font_siyuan_cn_bold)
        self.__title_tag_label.setStyleSheet(settings['title_tag_label_ss'])

        self.__title_line_label = QLabel(title_group)
        self.__title_line_label.setGeometry(8, 41, 402, 3)
        self.__title_line_label.setPixmap(QPixmap(settings['title_line']))
        # title_group END

        self.__tip_tag_label = QLabel('暂无等待任务', self)
        self.__tip_tag_label.setGeometry(32, 190, 386, 18)
        self.__tip_tag_label.setFont(font_siyuan_cn_regular)
        self.__tip_tag_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.__tip_tag_label.setStyleSheet(settings['tip_tag_label_ss'])

        self.__scroll_container = ScrollContainer()
        self.__scroll_container.setFixedSize(0, 0)

        self.__scroll_area = ScrollAreaWithShift(self)
        self.__scroll_area.setGeometry(32, 74, 386 + 24, 84 * 3)  # 24 = 14+8+1*2 = gap + bar_width + bar_border * 2
        self.__scroll_area.setWidget(self.__scroll_container)
        self.__scroll_area.setFrameShape(ScrollAreaWithShift.Shape.NoFrame)
        self.__scroll_area.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.__scroll_area.setStyleSheet(f'''
            {type(self.__scroll_area).__name__} {{ background: transparent;}}
            {settings['scroll_bar_ss']}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; background: none;}}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none;}}
        ''')

        self.__settings = settings

        # self.__scroll_test(5)

    def add_task(self, task_name: str, start_time: str):
        item = TaskListWidget.ScrollItemWidget(self.__settings['scroll_item'], task_name, start_time)
        item.setFixedSize(386, 84)
        self.__scroll_container.addItem(item)
        self.__tip_tag_label.setVisible(False)

    def remove_task(self, index: int):
        self.__scroll_container.removeItem(index)
        if self.__scroll_container.getItemNum() == 0:
            self.__tip_tag_label.setVisible(True)

    def __scroll_test(self, item_num: int):
        for i in range(item_num):
            self.add_task(f'视频识别任务{i}', '2025-12-31 23:23:23')
