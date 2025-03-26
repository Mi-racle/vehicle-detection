import argparse
import os

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu_mem', type=int, default=500)
    parser.add_argument('--warmup', type=bool, default=False)

    # params for text detector
    parser.add_argument('--det_algorithm', type=str, default='DB')
    parser.add_argument('--det_model_path', type=str)
    parser.add_argument('--det_limit_side_len', type=float, default=960)
    parser.add_argument('--det_limit_type', type=str, default='max')

    # DB params
    parser.add_argument('--det_db_thresh', type=float, default=0.3)
    parser.add_argument('--det_db_box_thresh', type=float, default=0.6)
    parser.add_argument('--det_db_unclip_ratio', type=float, default=1.5)
    parser.add_argument('--max_batch_size', type=int, default=10)
    parser.add_argument('--use_dilation', type=bool, default=False)
    parser.add_argument('--det_db_score_mode', type=str, default='fast')

    # EAST params
    parser.add_argument('--det_east_score_thresh', type=float, default=0.8)
    parser.add_argument('--det_east_cover_thresh', type=float, default=0.1)
    parser.add_argument('--det_east_nms_thresh', type=float, default=0.2)

    # SAST params
    parser.add_argument('--det_sast_score_thresh', type=float, default=0.5)
    parser.add_argument('--det_sast_nms_thresh', type=float, default=0.2)
    parser.add_argument('--det_sast_polygon', type=bool, default=False)

    # PSE params
    parser.add_argument('--det_pse_thresh', type=float, default=0)
    parser.add_argument('--det_pse_box_thresh', type=float, default=0.85)
    parser.add_argument('--det_pse_min_area', type=float, default=16)
    parser.add_argument('--det_pse_box_type', type=str, default='box')
    parser.add_argument('--det_pse_scale', type=int, default=1)

    # FCE params
    parser.add_argument('--scales', type=str, default='8, 16, 32')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--fourier_degree', type=int, default=5)
    parser.add_argument('--det_fce_box_type', type=str, default='poly')

    # params for text recognizer
    parser.add_argument('--rec_algorithm', type=str, default='CRNN')
    parser.add_argument('--rec_model_path', type=str)
    parser.add_argument('--rec_image_inverse', type=bool, default=True)
    parser.add_argument('--rec_image_shape', type=str, default='3, 48, 320')
    parser.add_argument('--rec_char_type', type=str, default='ch')
    parser.add_argument('--rec_batch_num', type=int, default=6)
    parser.add_argument('--max_text_length', type=int, default=25)

    parser.add_argument('--use_space_char', type=bool, default=True)
    parser.add_argument('--drop_score', type=float, default=0.5)
    parser.add_argument('--limited_max_width', type=int, default=1280)
    parser.add_argument('--limited_min_width', type=int, default=16)

    parser.add_argument(
        '--vis_font_path',
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'doc/fonts/simfang.ttf'
        )
    )
    parser.add_argument(
        '--rec_char_dict_path',
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'dicts/ppocr_keys_v1.txt'
        )
    )

    # params for text classifier
    parser.add_argument('--use_angle_cls', type=bool, default=False)
    parser.add_argument('--cls_model_path', type=str, default=None)
    parser.add_argument('--cls_image_shape', type=str, default='3, 48, 192')
    parser.add_argument('--label_list', type=list, default=['0', '180'])
    parser.add_argument('--cls_batch_num', type=int, default=6)
    parser.add_argument('--cls_thresh', type=float, default=0.9)

    parser.add_argument('--enable_mkldnn', type=bool, default=False)
    parser.add_argument('--use_pdserving', type=bool, default=False)

    # params .yaml
    parser.add_argument(
        '--det_yaml_path',
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'configs/ch_PP-OCRv4_det.yaml'
        )
    )
    parser.add_argument(
        '--rec_yaml_path',
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'configs/ch_PP-OCRv4_rec.yaml'
        )
    )
    parser.add_argument(
        '--cls_yaml_path',
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'configs/cls_mv3.yaml'
        )
    )

    # multi-process
    parser.add_argument('--use_mp', type=bool, default=False)
    parser.add_argument('--total_process_num', type=int, default=1)
    parser.add_argument('--process_id', type=int, default=0)

    parser.add_argument('--benchmark', type=bool, default=False)
    parser.add_argument('--save_log_path', type=str, default='./log_output/')

    parser.add_argument('--show_log', type=bool, default=False)

    return parser.parse_args()


def read_network_config_from_yaml(yaml_path, char_num=None):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError('{} is not existed.'.format(yaml_path))

    with open(yaml_path, encoding='utf-8') as f:
        res = yaml.safe_load(f)

    if res.get('Architecture') is None:
        raise ValueError('{} has no Architecture'.format(yaml_path))

    if res['Architecture']['Head']['name'] == 'MultiHead' and char_num is not None:
        res['Architecture']['Head']['out_channels_list'] = {
            'CTCLabelDecode': char_num,
            'SARLabelDecode': char_num + 2,
            'NRTRLabelDecode': char_num + 3
        }

    return res['Architecture']


def analysis_config(weights_path, yaml_path=None, char_num=None):
    abs_path = os.path.abspath(weights_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f'{abs_path} is not found.')

    if yaml_path is not None:
        return read_network_config_from_yaml(yaml_path, char_num=char_num)

    weights_basename = os.path.basename(weights_path)
    weights_name = weights_basename.lower()

    if weights_name == 'ch_ptocr_server_v2.0_det_infer.pth':
        network_config = {'model_type': 'det',
                          'algorithm': 'DB',
                          'Transform': None,
                          'Backbone': {'name': 'ResNet_vd', 'layers': 18, 'disable_se': True},
                          'Neck': {'name': 'DBFPN', 'out_channels': 256},
                          'Head': {'name': 'DBHead', 'k': 50}}

    elif weights_name == 'ch_ptocr_server_v2.0_rec_infer.pth':
        network_config = {'model_type': 'rec',
                          'algorithm': 'CRNN',
                          'Transform': None,
                          'Backbone': {'name': 'ResNet', 'layers': 34},
                          'Neck': {'name': 'SequenceEncoder', 'hidden_size': 256, 'encoder_type': 'rnn'},
                          'Head': {'name': 'CTCHead', 'fc_decay': 4e-05}}

    elif weights_name in ['ch_ptocr_mobile_v2.0_det_infer.pth']:
        network_config = {'model_type': 'det',
                          'algorithm': 'DB',
                          'Transform': None,
                          'Backbone': {'name': 'MobileNetV3', 'model_name': 'large', 'scale': 0.5, 'disable_se': True},
                          'Neck': {'name': 'DBFPN', 'out_channels': 96},
                          'Head': {'name': 'DBHead', 'k': 50}}

    elif weights_name == 'ch_ptocr_mobile_v2.0_rec_infer.pth':
        network_config = {'model_type': 'rec',
                          'algorithm': 'CRNN',
                          'Transform': None,
                          'Backbone': {'model_name': 'small', 'name': 'MobileNetV3', 'scale': 0.5,
                                       'small_stride': [1, 2, 2, 2]},
                          'Neck': {'name': 'SequenceEncoder', 'hidden_size': 48, 'encoder_type': 'rnn'},
                          'Head': {'name': 'CTCHead', 'fc_decay': 4e-05}}

    elif weights_name == 'ch_ptocr_mobile_v2.0_cls_infer.pth':
        network_config = {'model_type': 'cls',
                          'algorithm': 'CLS',
                          'Transform': None,
                          'Backbone': {'name': 'MobileNetV3', 'model_name': 'small', 'scale': 0.35},
                          'Neck': None,
                          'Head': {'name': 'ClsHead', 'class_dim': 2}}

    elif weights_name == 'ch_ptocr_v2_rec_infer.pth':
        network_config = {'model_type': 'rec',
                          'algorithm': 'CRNN',
                          'Transform': None,
                          'Backbone': {'name': 'MobileNetV1Enhance', 'scale': 0.5},
                          'Neck': {'name': 'SequenceEncoder', 'hidden_size': 64, 'encoder_type': 'rnn'},
                          'Head': {'name': 'CTCHead', 'mid_channels': 96, 'fc_decay': 2e-05}}

    elif weights_name == 'ch_ptocr_v2_det_infer.pth':
        network_config = {'model_type': 'det',
                          'algorithm': 'DB',
                          'Transform': None,
                          'Backbone': {'name': 'MobileNetV3', 'model_name': 'large', 'scale': 0.5, 'disable_se': True},
                          'Neck': {'name': 'DBFPN', 'out_channels': 96},
                          'Head': {'name': 'DBHead', 'k': 50}}

    elif weights_name == 'ch_ptocr_v3_rec_infer.pth':
        network_config = {'model_type': 'rec',
                          'algorithm': 'CRNN',
                          'Transform': None,
                          'Backbone': {'name': 'MobileNetV1Enhance',
                                       'scale': 0.5,
                                       'last_conv_stride': [1, 2],
                                       'last_pool_type': 'avg'},
                          'Neck': {'name': 'SequenceEncoder',
                                   'dims': 64,
                                   'depth': 2,
                                   'hidden_dims': 120,
                                   'use_guide': True,
                                   'encoder_type': 'svtr'},
                          'Head': {'name': 'CTCHead', 'fc_decay': 2e-05}
                          }

    elif weights_name == 'ch_ptocr_v3_det_infer.pth':
        network_config = {'model_type': 'det',
                          'algorithm': 'DB',
                          'Transform': None,
                          'Backbone': {'name': 'MobileNetV3', 'model_name': 'large', 'scale': 0.5, 'disable_se': True},
                          'Neck': {'name': 'RSEFPN', 'out_channels': 96, 'shortcut': True},
                          'Head': {'name': 'DBHead', 'k': 50}}

    elif weights_name == 'det_mv3_db_v2.0_infer.pth':
        network_config = {'model_type': 'det',
                          'algorithm': 'DB',
                          'Transform': None,
                          'Backbone': {'name': 'MobileNetV3', 'model_name': 'large'},
                          'Neck': {'name': 'DBFPN', 'out_channels': 256},
                          'Head': {'name': 'DBHead', 'k': 50}}

    elif weights_name == 'det_r50_vd_db_v2.0_infer.pth':
        network_config = {'model_type': 'det',
                          'algorithm': 'DB',
                          'Transform': None,
                          'Backbone': {'name': 'ResNet_vd', 'layers': 50},
                          'Neck': {'name': 'DBFPN', 'out_channels': 256},
                          'Head': {'name': 'DBHead', 'k': 50}}

    elif weights_name == 'det_mv3_east_v2.0_infer.pth':
        network_config = {'model_type': 'det',
                          'algorithm': 'EAST',
                          'Transform': None,
                          'Backbone': {'name': 'MobileNetV3', 'model_name': 'large'},
                          'Neck': {'name': 'EASTFPN', 'model_name': 'small'},
                          'Head': {'name': 'EASTHead', 'model_name': 'small'}}

    elif weights_name == 'det_r50_vd_east_v2.0_infer.pth':
        network_config = {'model_type': 'det',
                          'algorithm': 'EAST',
                          'Transform': None,
                          'Backbone': {'name': 'ResNet_vd', 'layers': 50},
                          'Neck': {'name': 'EASTFPN', 'model_name': 'large'},
                          'Head': {'name': 'EASTHead', 'model_name': 'large'}}

    elif weights_name == 'det_r50_vd_sast_icdar15_v2.0_infer.pth':
        network_config = {'model_type': 'det',
                          'algorithm': 'SAST',
                          'Transform': None,
                          'Backbone': {'name': 'ResNet_SAST', 'layers': 50},
                          'Neck': {'name': 'SASTFPN', 'with_cab': True},
                          'Head': {'name': 'SASTHead'}}

    elif weights_name == 'det_r50_vd_sast_totaltext_v2.0_infer.pth':
        network_config = {'model_type': 'det',
                          'algorithm': 'SAST',
                          'Transform': None,
                          'Backbone': {'name': 'ResNet_SAST', 'layers': 50},
                          'Neck': {'name': 'SASTFPN', 'with_cab': True},
                          'Head': {'name': 'SASTHead'}}

    elif weights_name == 'en_server_pgneta_infer.pth':
        network_config = {'model_type': 'e2e',
                          'algorithm': 'PGNet',
                          'Transform': None,
                          'Backbone': {'name': 'ResNet', 'layers': 50},
                          'Neck': {'name': 'PGFPN'},
                          'Head': {'name': 'PGHead'}}

    elif weights_name == 'en_ptocr_mobile_v2.0_table_det_infer.pth':
        network_config = {'model_type': 'det', 'algorithm': 'DB',
                          'Transform': None,
                          'Backbone': {'name': 'MobileNetV3', 'model_name': 'large', 'scale': 0.5, 'disable_se': False},
                          'Neck': {'name': 'DBFPN', 'out_channels': 96},
                          'Head': {'name': 'DBHead', 'k': 50}}

    elif weights_name == 'en_ptocr_mobile_v2.0_table_rec_infer.pth':
        network_config = {'model_type': 'rec',
                          'algorithm': 'CRNN',
                          'Transform': None,
                          'Backbone': {'model_name': 'large', 'name': 'MobileNetV3', },
                          'Neck': {'name': 'SequenceEncoder', 'hidden_size': 96, 'encoder_type': 'rnn'},
                          'Head': {'name': 'CTCHead', 'fc_decay': 4e-05}}

    elif 'om_' in weights_name and '_rec_' in weights_name:
        network_config = {'model_type': 'rec',
                          'algorithm': 'CRNN',
                          'Transform': None,
                          'Backbone': {'model_name': 'small', 'name': 'MobileNetV3', 'scale': 0.5,
                                       'small_stride': [1, 2, 2, 2]},
                          'Neck': {'name': 'SequenceEncoder', 'hidden_size': 48, 'encoder_type': 'om'},
                          'Head': {'name': 'CTCHead', 'fc_decay': 4e-05}}

    else:
        network_config = {'model_type': 'rec',
                          'algorithm': 'CRNN',
                          'Transform': None,
                          'Backbone': {'model_name': 'small', 'name': 'MobileNetV3', 'scale': 0.5,
                                       'small_stride': [1, 2, 2, 2]},
                          'Neck': {'name': 'SequenceEncoder', 'hidden_size': 48, 'encoder_type': 'rnn'},
                          'Head': {'name': 'CTCHead', 'fc_decay': 4e-05}}

    return network_config
