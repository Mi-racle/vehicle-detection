"""
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import sys
import cv2
import numpy as np


class DecodeImage:
    """ decode image """
    def __init__(self, img_mode='RGB', channel_first=False, **kwargs):
        self.img_mode = img_mode
        self.channel_first = channel_first

    def __call__(self, data):
        img = data['image']
        assert type(img) is bytes and len(img) > 0, "invalid input 'img' in DecodeImage"
        img = np.frombuffer(img, dtype='uint8')
        img = cv2.imdecode(img, 1)
        if img is None:
            return None
        if self.img_mode == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif self.img_mode == 'RGB':
            assert img.shape[2] == 3, 'invalid shape of image[%s]' % (img.shape)
            img = img[:, :, ::-1]

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        data['image'] = img
        return data


class NRTRDecodeImage:
    """ decode image """
    def __init__(self, img_mode='RGB', channel_first=False, **kwargs):
        self.img_mode = img_mode
        self.channel_first = channel_first

    def __call__(self, data):
        img = data['image']
        assert type(img) is bytes and len(img) > 0, "invalid input 'img' in DecodeImage"
        img = np.frombuffer(img, dtype='uint8')

        img = cv2.imdecode(img, 1)

        if img is None:
            return None
        if self.img_mode == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif self.img_mode == 'RGB':
            assert img.shape[2] == 3, 'invalid shape of image[%s]' % (img.shape)
            img = img[:, :, ::-1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.channel_first:
            img = img.transpose((2, 0, 1))
        data['image'] = img
        return data


class NormalizeImage:
    """
    normalize image such as substract mean, divide std
    """
    def __init__(self, scale=None, mean=None, std=None, order='chw', **kwargs):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, data):
        img = data['image']
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)
        assert isinstance(img,
                          np.ndarray), "invalid input 'img' in NormalizeImage"
        data['image'] = (
            img.astype('float32') * self.scale - self.mean) / self.std
        return data


class ToCHWImage:
    """
    convert hwc image to chw image
    """
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        img = data['image']
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)
        data['image'] = img.transpose((2, 0, 1))
        return data


# class Fasttext:
#     def __init__(self, path="None", **kwargs):
#         import fasttext
#         self.fast_model = fasttext.load_model(path)
#
#     def __call__(self, data):
#         label = data['label']
#         fast_label = self.fast_model[label]
#         data['fast_label'] = fast_label
#         return data


class KeepKeys:
    def __init__(self, keep_keys, **kwargs):
        self.keep_keys = keep_keys

    def __call__(self, data):
        data_list = []
        for key in self.keep_keys:
            data_list.append(data[key])
        return data_list


class Resize:
    def __init__(self, size=(640, 640), **kwargs):
        self.size = size

    def resize_image(self, img):
        resize_h, resize_w = self.size
        ori_h, ori_w = img.shape[:2]  # (h, w, c)
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        return img, [ratio_h, ratio_w]

    def __call__(self, data):
        img = data['image']
        text_polys = data['polys']

        img_resize, [ratio_h, ratio_w] = self.resize_image(img)
        new_boxes = []
        for box in text_polys:
            new_box = []
            for cord in box:
                new_box.append([cord[0] * ratio_w, cord[1] * ratio_h])
            new_boxes.append(new_box)
        data['image'] = img_resize
        data['polys'] = np.array(new_boxes, dtype=np.float32)
        return data


class DetResizeForTest:
    def __init__(self, **kwargs):
        super(DetResizeForTest, self).__init__()
        self.resize_type = 0
        if 'image_shape' in kwargs:
            self.image_shape = kwargs['image_shape']
            self.resize_type = 1
        elif 'limit_side_len' in kwargs:
            self.limit_side_len = kwargs['limit_side_len']
            self.limit_type = kwargs.get('limit_type', 'min')
        elif 'resize_long' in kwargs:
            self.resize_type = 2
            self.resize_long = kwargs.get('resize_long', 960)
        else:
            self.limit_side_len = 736
            self.limit_type = 'min'

    def __call__(self, data):
        img = data['image']
        src_h, src_w, _ = img.shape

        if self.resize_type == 0:
            # img, shape = self.resize_image_type0(img)
            img, [ratio_h, ratio_w] = self.resize_image_type0(img)
        elif self.resize_type == 2:
            img, [ratio_h, ratio_w] = self.resize_image_type2(img)
        else:
            # img, shape = self.resize_image_type1(img)
            img, [ratio_h, ratio_w] = self.resize_image_type1(img)
        data['image'] = img
        data['shape'] = np.array([src_h, src_w, ratio_h, ratio_w])
        return data

    def resize_image_type1(self, img):
        resize_h, resize_w = self.image_shape
        ori_h, ori_w = img.shape[:2]  # (h, w, c)
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        # return img, np.array([ori_h, ori_w])
        return img, [ratio_h, ratio_w]

    def resize_image_type0(self, img):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        limit_side_len = self.limit_side_len
        h, w, c = img.shape

        # limit the max side
        if self.limit_type == 'max':
            if max(h, w) > limit_side_len:
                if h > w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        elif self.limit_type == 'min':
            if min(h, w) < limit_side_len:
                if h < w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        elif self.limit_type == 'resize_long':
            ratio = float(limit_side_len) / max(h, w)
        else:
            raise Exception('not support limit type, image ')
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)

        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            img = cv2.resize(img, (int(resize_w), int(resize_h)))
        except:
            print(img.shape, resize_w, resize_h)
            sys.exit(0)
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return img, [ratio_h, ratio_w]

    def resize_image_type2(self, img):
        h, w, _ = img.shape

        resize_w = w
        resize_h = h

        if resize_h > resize_w:
            ratio = float(self.resize_long) / resize_h
        else:
            ratio = float(self.resize_long) / resize_w

        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return img, [ratio_h, ratio_w]


class E2EResizeForTest:
    def __init__(self, **kwargs):
        super(E2EResizeForTest, self).__init__()
        self.max_side_len = kwargs['max_side_len']
        self.valid_set = kwargs['valid_set']

    def __call__(self, data):
        img = data['image']
        src_h, src_w, _ = img.shape
        if self.valid_set == 'totaltext':
            im_resized, [ratio_h, ratio_w] = self.resize_image_for_totaltext(
                img, max_side_len=self.max_side_len)
        else:
            im_resized, (ratio_h, ratio_w) = self.resize_image(
                img, max_side_len=self.max_side_len)
        data['image'] = im_resized
        data['shape'] = np.array([src_h, src_w, ratio_h, ratio_w])
        return data

    def resize_image_for_totaltext(self, im, max_side_len=512):

        h, w, _ = im.shape
        resize_w = w
        resize_h = h
        ratio = 1.25
        if h * ratio > max_side_len:
            ratio = float(max_side_len) / resize_h
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        im = cv2.resize(im, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return im, (ratio_h, ratio_w)

    def resize_image(self, im, max_side_len=512):
        """
        resize image to a size multiple of max_stride which is required by the network
        :param im: the resized image
        :param max_side_len: limit of max image size to avoid out of memory in gpu
        :return: the resized image and the resize ratio
        """
        h, w, _ = im.shape

        resize_w = w
        resize_h = h

        # Fix the longer side
        if resize_h > resize_w:
            ratio = float(max_side_len) / resize_h
        else:
            ratio = float(max_side_len) / resize_w

        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        im = cv2.resize(im, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return im, (ratio_h, ratio_w)


class KieResize:
    def __init__(self, **kwargs):
        super(KieResize, self).__init__()
        self.max_side, self.min_side = kwargs['img_scale'][0], kwargs[
            'img_scale'][1]

    def __call__(self, data):
        img = data['image']
        points = data['points']
        src_h, src_w, _ = img.shape
        im_resized, scale_factor, [ratio_h, ratio_w
                                   ], [new_h, new_w] = self.resize_image(img)
        resize_points = self.resize_boxes(img, points, scale_factor)
        data['ori_image'] = img
        data['ori_boxes'] = points
        data['points'] = resize_points
        data['image'] = im_resized
        data['shape'] = np.array([new_h, new_w])
        return data

    @staticmethod
    def resize_image(img):
        norm_img = np.zeros([1024, 1024, 3], dtype='float32')
        scale = [512, 1024]
        h, w = img.shape[:2]
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
        resize_w, resize_h = int(w * float(scale_factor) + 0.5), int(h * float(
            scale_factor) + 0.5)
        max_stride = 32
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        im = cv2.resize(img, (resize_w, resize_h))
        new_h, new_w = im.shape[:2]
        w_scale = new_w / w
        h_scale = new_h / h
        scale_factor = np.array(
            [w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        norm_img[:new_h, :new_w, :] = im
        return norm_img, scale_factor, [h_scale, w_scale], [new_h, new_w]

    @staticmethod
    def resize_boxes(im, points, scale_factor):
        points = points * scale_factor
        img_shape = im.shape[:2]
        points[:, 0::2] = np.clip(points[:, 0::2], 0, img_shape[1])
        points[:, 1::2] = np.clip(points[:, 1::2], 0, img_shape[0])
        return points


def perform_operators(data, ops=None):
    """ transform """
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


def create_operators(op_param_list: list, global_config=None):
    """
    create operators based on the config
    Args:
        op_param_list(list): a dict list, used to create some operators
        global_config: configs
    """
    ops = []

    if not op_param_list:
        return ops

    for operator in op_param_list:
        assert isinstance(operator, dict) and len(operator) == 1, 'yaml format error'
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = eval(op_name)(**param)
        ops.append(op)

    return ops
