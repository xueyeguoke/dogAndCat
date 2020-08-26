import cv2 as cv
import os
import numpy as np


def main():
    # print(find_min_size(path))#找到最小的图片用来确定图片reshze目标

    path = r'G:\BaiduNetdiskDownload\dogandcat\train\train'
    pretreat_imges(path, 'datas/train_reshape', (128, 128))
    path = r'G:\BaiduNetdiskDownload\dogandcat\test\test'
    pretreat_imges(path, 'datas/test_reshape', (128, 128))


def create_dir_try(path):
    '''
    创建文件夹path
    :param path:
    :return:
    '''
    if not os.path.exists(path):
        os.makedirs(path)


def pretreat_imges(path, save_path, shape):
    '''
    缩放图片
    :param path:
    :param save_path:
    :param shape:
    :return:
    '''
    create_dir_try(save_path)
    files = os.listdir(path)
    for i in range(len(files)):
        img = cv.imread(os.path.join(path, files[i]))
        img = cv.resize(img, shape)

        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # 高斯模糊
        img_gaussiian = cv.GaussianBlur(img_gray, ksize=(5, 5), sigmaX=1, sigmaY=1)
        kernel = np.asarray([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ])
        sobely = cv.filter2D(img_gaussiian, ddepth=6, kernel=kernel)
        sobelx = cv.filter2D(img_gaussiian, ddepth=6, kernel=kernel.T)
        sobely = (sobely + 256) / 512
        r, g, b = cv.split(img)
        sobely = np.uint8(sobely)
        sobelx = np.uint8(sobelx)
        img_rst = cv.merge((r, g, b, sobelx, sobely))
        file_path = os.path.join(save_path, files[i]) + '.npy'
        np.save(file_path, img_rst)

        # cv.imwrite(os.path.join(save_path, files[i]), img)
        if i % 1000 == 0:
            print('第{}个图片,共{}个'.format(i, len(files)))
    print('完成')


def find_min_size(path):
    '''
    寻找最小的图片大小，用来确定图片需要缩放到多大比较合适
    :param path:
    :return:
    '''
    files = os.listdir(path)
    # 找到最小的图片
    x = 1000
    y = 1000
    for i in range(len(files)):
        img_path = os.path.join(path, files[i])
        img = cv.imread(img_path, 1)
        if x > np.shape(img)[0]:
            x = np.shape(img)[0]
            print('*' * 100)
            print('{},x={}'.format(files[i], x))
        if y > np.shape(img)[1]:
            y = np.shape(img)[1]
            print('*' * 100)
            print('{},y={}'.format(files[i], y))
        if i % 1000 == 0:
            print(i, x, y)
    return (x, y)


if __name__ == '__main__':
    main()
