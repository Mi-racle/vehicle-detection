import math

from PIL import Image, ImageDraw, ImageFont

from ocr.operators import *


def get_font_size(font, text):
    left, top, right, bottom = font.getbbox(text)
    return right - left, bottom - top


def draw_text_det_res(dt_boxes, img_path):
    src_im = cv2.imread(img_path)

    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)

    return src_im


def draw_ocr_box_txt(
        image,
        boxes,
        txts,
        scores=None,
        drop_score=0.5,
        font_path='./doc/simfang.ttf'
):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    import random

    random.seed(0)

    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)

    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue

        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )

        draw_left.polygon(box, fill=color)
        draw_right.polygon(
            [box[0][0], box[0][1], box[1][0], box[1][1], box[2][0], box[2][1], box[3][0], box[3][1]],
            outline=color
        )
        box_height = math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)
        box_width = math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)

        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding='utf-8')
            cur_y = box[0][1]

            for c in txt:
                char_size = get_font_size(font, c)
                draw_right.text((box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]

        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding='utf-8')
            draw_right.text((box[0][0], box[0][1]), txt, fill=(0, 0, 0), font=font)

    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))

    return np.array(img_show)
