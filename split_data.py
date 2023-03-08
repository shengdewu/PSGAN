import os
import random


if __name__ == '__main__':
    image_dir_name = "images"
    seg_dir_name = "segs"
    landmark_dir_name = "landmarks"
    makeup = "makeup.txt"
    non_makeup = "non-makeup.txt"

    in_root = '/mnt/sda2/makeup.data/MT-Dataset'
    makeup_name = [f'makeup/{name}' for name in os.listdir(f'{in_root}/{image_dir_name}/makeup')]
    nonmakeup_name = [f'non-makeup/{name}' for name in os.listdir(f'{in_root}/{image_dir_name}/non-makeup')]

    ps_makeup_name = [name for name in makeup_name if os.path.exists(f'{in_root}/{seg_dir_name}/{name}') and os.path.exists(f'{in_root}/{landmark_dir_name}/{name}')]
    ps_nonmakeup_name = [name for name in nonmakeup_name if os.path.exists(f'{in_root}/{seg_dir_name}/{name}') and os.path.exists(f'{in_root}/{landmark_dir_name}/{name}')]

    sc_makeup_name = [name for name in makeup_name if os.path.exists(f'{in_root}/{seg_dir_name}/{name}')]
    sc_nonmakeup_name = [name for name in nonmakeup_name if os.path.exists(f'{in_root}/{seg_dir_name}/{name}')]

    test_makeup_name = random.sample(ps_makeup_name, 20)
    test_nonmakeup_name = random.sample(ps_nonmakeup_name, 10)


    def write(file, names, skip_names=None):
        with open(file, mode='w') as f:
            for name in names:
                if skip_names is not None and name in skip_names:
                    continue
                f.write(f'{name}\n')


    write(f'{in_root}/psgan-makeup.txt', ps_makeup_name, test_makeup_name)
    write(f'{in_root}/psgan-non-makeup.txt', ps_nonmakeup_name, test_nonmakeup_name)
    write(f'{in_root}/scgan-makeup.txt', sc_makeup_name, test_makeup_name)
    write(f'{in_root}/scgan-non-makeup.txt', sc_nonmakeup_name, test_nonmakeup_name)
    write(f'{in_root}/test-makeup.txt', test_makeup_name)
    write(f'{in_root}/test-non-makeup.txt', test_nonmakeup_name)