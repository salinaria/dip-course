import cv2
import numpy as np
import matplotlib.pyplot as plt


def showInfo(img, name):
    print(name)
    print(img.shape)
    print(img.dtype)
    print(np.min(img), np.max(img))
    print('')


spine = cv2.imread("spineXray.tif", cv2.IMREAD_GRAYSCALE)
chest = cv2.imread("chest.tif", cv2.IMREAD_ANYDEPTH)

showInfo(spine, 'Spine')
showInfo(chest, 'Chest')


def performCLAHE(img):
    clahe = cv2.createCLAHE(clipLimit = 20)
    final_img = clahe.apply(img)
    return final_img


def transform(img, bit_depth):
    ans = np.zeros(img.shape)
    pdf = {}
    new_colors = {}
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            try:
                pdf[int(img[i][j])] += 1
            except:
                pdf[int(img[i][j])] = 1
    for i in range(2**bit_depth):
        if i == 0:
            new_colors[0] = pdf[0]
        else:
            if i in pdf:
                new_colors[i] = new_colors[i-1] + pdf[i]
            else:
                new_colors[i] = new_colors[i-1]

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            sigma = new_colors[img[i][j]]
            ans[i][j] = (2**bit_depth-1)*sigma/(img.shape[0]*img.shape[1])

    return ans



clahe_spine = performCLAHE(spine)
hist_spine = transform(spine, 8)

plt.subplot(3,3,1)
plt.title('Original')
plt.imshow(spine, cmap="gray", vmin=0, vmax=255)
plt.axis('off')
plt.tight_layout()

plt.subplot(3,3,2)
plt.title('CLAHE')
plt.imshow(clahe_spine, cmap="gray", vmin=0, vmax=255)
plt.axis('off')
plt.tight_layout()

plt.subplot(3,3,3)
plt.title('Hist equalization')
plt.imshow(hist_spine, cmap="gray", vmin=0, vmax=255)
plt.axis('off')
plt.tight_layout()

plt.subplot(3,3,4)
plt.ylabel('Histogram')
plt.hist(spine.flat, bins=[i for i in range(0, 256, 4)], density=True)
plt.tight_layout()

plt.subplot(3,3,5)
plt.hist(clahe_spine.flat, bins=[i for i in range(0, 256, 4)], density=True)
plt.tight_layout()

plt.subplot(3,3,6)
plt.hist(hist_spine.flat, bins=[i for i in range(0, 256, 4)], density=True)
plt.tight_layout()

plt.subplot(3,3,7)
plt.ylabel('Cumulative')
plt.hist(spine.flat, bins=[i for i in range(
    0, 256, 4)], density=True, cumulative=True)
plt.tight_layout()

plt.subplot(3,3,8)
plt.hist(clahe_spine.flat, bins=[i for i in range(
    0, 256, 4)], density=True, cumulative=True)
plt.tight_layout()

plt.subplot(3,3,9)
plt.hist(hist_spine.flat, bins=[i for i in range(
    0, 256, 4)], density=True, cumulative=True)
plt.tight_layout()
plt.savefig('./res_q4_spine',cmap="gray")
plt.show()



clahe_chest = performCLAHE(chest)
hist_chest = transform(chest, 16)

plt.subplot(3,3,1)
plt.title("Original")
plt.imshow(chest, cmap="gray")
plt.axis('off')

plt.subplot(3,3,2)
plt.title('CLAHE')
plt.imshow(clahe_chest, cmap="gray")
plt.axis('off')

plt.subplot(3,3,3)
plt.title('Hist equalization')
plt.imshow(hist_chest, cmap="gray")
plt.axis('off')

plt.subplot(3,3,4)
plt.ylabel('Histogram')
plt.hist(chest.flat, bins=[i for i in range(0, 2**16, 1024)], density=True)
plt.tight_layout()

plt.subplot(3,3,5)
plt.hist(clahe_chest.flat, bins=[i for i in range(0, 2**16, 1024)], density=True)
plt.tight_layout()

plt.subplot(3,3,6)
plt.hist(hist_chest.flat, bins=[i for i in range(0, 2**16, 1024)], density=True)
plt.tight_layout()

plt.subplot(3,3,7)
plt.ylabel('Cumulative')
plt.hist(chest.flat, bins=[i for i in range(
    0, 2**16, 1024)], density=True, cumulative=True)
plt.tight_layout()

plt.subplot(3,3,8)
plt.hist(clahe_chest.flat, bins=[i for i in range(
    0, 2**16, 1024)], density=True, cumulative=True)
plt.tight_layout()

plt.subplot(3,3,9)
plt.hist(hist_chest.flat, bins=[i for i in range(
    0, 2**16, 1024)], density=True, cumulative=True)
plt.tight_layout()
plt.savefig('./res_q4_chest',cmap="gray")
plt.show()
