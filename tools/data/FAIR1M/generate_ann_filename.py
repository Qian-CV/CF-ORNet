import os


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def write_file(file_dir):
    # mkdir(os.path.join(file_dir, 'train.txt'))
    f = open(os.path.join(file_dir, 'train.txt'), 'w')
    for num in range(8287):
        f.write(f'{num}\n')
    f.close()


if __name__ == '__main__':
    write_file('/media/ubuntu/CE425F4D425F3983/datasets/FAIR1M2.0/validation/')
