import os

if __name__ == '__main__':
    dataset_path = '../../resources/simplified-car-color-dataset'
    src_sets = ['test', 'train', 'val']
    src_cls = 'orange'

    for src_set in src_sets:
        src_path = os.path.join(dataset_path, src_set, src_cls)

        if not os.path.exists(src_path):
            print('Non-existent class')
            continue

        for filename in os.listdir(src_path):
            src_file = os.path.join(src_path, filename)
            os.remove(src_file)

        os.removedirs(src_path)
