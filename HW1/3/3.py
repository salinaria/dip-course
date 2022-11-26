import matplotlib.pyplot as plt
import numpy as np
import math

img = plt.imread('./3/transformed.png')

plt.imshow(img, cmap='gray')
plt.show()


def backto512():
    matrix = np.linalg.inv(
        np.array([[512/383, 0, 0], [0, 512/255, 0], [0, 0, 1]]))
    new_img = np.ones(img.shape)

    for i in range(512):
        for j in range(512):
            temp = np.uint(np.dot(matrix, np.array([i, j, 1])))
            new_img[i][j] = img[temp[0]][temp[1]]

    plt.imshow(new_img, cmap='gray')
    plt.savefig('./3/512.jpg', cmap='gray')
    plt.show()
    return new_img


new_img = backto512()


def padimg():
    new_big_img = np.pad(new_img, ((256, 256), (256, 256)))
    plt.imshow(new_big_img, cmap='gray')
    plt.savefig('./3/1024.jpg')
    plt.show()
    return new_big_img


new_big_img = padimg()


def translate():
    matrix = np.array([[1, 0, -100], [0, 1, 250], [0, 0, 1]])
    new_img = np.zeros((1024, 1024))

    for i in range(256, 768):
        for j in range(256, 768):
            new_cor = np.int64(np.dot(matrix, np.array([i, j, 1])))
            new_img[new_cor[0]][new_cor[1]] = new_big_img[i][j]
            
    plt.imshow(new_img, cmap='gray')
    plt.savefig('./3/translate.jpg', cmap='gray')
    plt.show()
    return new_img


translated = translate()


def shear():
    matrix = np.array([[1, -0.3, 0], [0, 1, 0], [0, 0, 1]])
    new_img = np.zeros((1024, 1024))

    for i in range(256, 768):
        for j in range(256, 768):
            new_cor = np.int64(np.dot(matrix, np.array([i, j, 1])))
            new_img[new_cor[0]][new_cor[1]] = new_big_img[i][j]
            
    plt.imshow(new_img, cmap='gray')
    plt.savefig('./3/shear.jpg', cmap='gray')
    plt.show()
    return new_img


sheared = shear()


def rotate():
    matrix = np.linalg.inv(np.array([[np.cos(math.radians(-20)), -1*np.sin(math.radians(-20)), 0], [
                         np.sin(math.radians(-20)), np.cos(math.radians(-20)), 0], [0, 0, 1]]))
    new_img = np.zeros((1024, 1024))

    for i in range(1024):
        for j in range(1024):
            new_cor = np.int64(np.dot(matrix, np.array([i, j, 1])))
            try:
                new_img[i][j] = new_big_img[new_cor[0]][new_cor[1]]
            except:
                hi = 1

    plt.imshow(new_img, cmap='gray')
    plt.savefig('./3/rotate.jpg', cmap='gray')
    plt.show()
    return new_img


rotated = rotate()
