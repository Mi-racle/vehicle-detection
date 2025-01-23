import os
from shutil import move

if __name__ == '__main__':
    dataset_path = '../../resources/simplified-car-color-dataset'
    src_sets = ['test', 'train', 'val']
    src_cls = 'gold'
    dst_cls = 'yellow'

    for src_set in src_sets:
        src_path = os.path.join(dataset_path, src_set, src_cls)
        dst_path = os.path.join(dataset_path, src_set, dst_cls)

        if not os.path.exists(src_path) or not os.path.exists(dst_path):
            print('Non-existent class')
            continue

        for filename in os.listdir(src_path):
            src_file = os.path.join(src_path, filename)
            dst_file = os.path.join(dst_path, filename)
            move(src_file, dst_file)

        os.removedirs(src_path)
