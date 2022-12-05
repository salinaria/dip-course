import cv2
import numpy as np
import matplotlib.pyplot as plt

# Part A : Implementaion cross correlation function with various filters

filters = {
    'mean': np.ones((3, 3)) / 9,
    'sobel_x': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
    'sobel_y': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    'laplacian': np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
    'mean5': np.ones((5,5)) / 25,
}

def cross_correlation(img, filter):
    output = np.zeros(img.shape)
    if filter == 'median':
        for i in range(1, output.shape[0] - 1):
            for j in range(1, output.shape[1] - 1):
                output[i, j] = np.median(img[i - 1: i + 2, j - 1: j + 2])
    elif filter == 'median5':
        for i in range(2, output.shape[0] - 2):
            for j in range(2, output.shape[1] - 2):
                output[i, j] = np.median(img[i - 2: i + 3, j - 2: j + 3])
    else:
        kernel = filters[filter]
        k_shape = kernel.shape
        r = int((k_shape[0] - 1) / 2)
        for i in range(r, output.shape[0] - r):
            for j in range(r, output.shape[1] - r):
                sum = 0
                for m in range(-r, r+1):
                    for n in range(-r, r+1):
                        sum += (img[i + m, j + n] * kernel[m + r, n + r])
                output[i, j] = sum
    return output

# Part B : Plot filters on MRI picture

mri = cv2.imread('MRI.png', cv2.IMREAD_GRAYSCALE)

mean_mri = cross_correlation(mri, 'mean')
sobelx_mri = cross_correlation(mri, 'sobel_x')
sobely_mri = cross_correlation(mri, 'sobel_y')
laplacian_mri = cross_correlation(mri, 'laplacian')
median_mri = cross_correlation(mri, 'median')

plt.subplot(2, 3,1)
plt.imshow(mri, cmap='gray')
plt.title('original')
plt.axis('off')
plt.tight_layout()

plt.subplot(2, 3,2)
plt.imshow(laplacian_mri, cmap='gray')
plt.title('laplacian')
plt.axis('off')
plt.tight_layout()

plt.subplot(2, 3,3)
plt.imshow(mean_mri, cmap='gray')
plt.title('mean')
plt.axis('off')
plt.tight_layout()

plt.subplot(2, 3,4)
plt.imshow(median_mri, cmap='gray')
plt.title('median')
plt.axis('off')
plt.tight_layout()

plt.subplot(2, 3,5)
plt.imshow(sobelx_mri, cmap='gray')
plt.title('sobel x')
plt.axis('off')
plt.tight_layout()

plt.subplot(2, 3,6)
plt.imshow(sobely_mri, cmap='gray')
plt.title('sobel y')
plt.axis('off')
plt.tight_layout()

plt.savefig('./filters')
plt.show()


# Part C : 5 * 5 mean and median filters
 
mean5_mri = cross_correlation(mri, 'mean5')
median5_mri = cross_correlation(mri, 'median5')

plt.subplot(1, 2,1)
plt.imshow(mean5_mri, cmap='gray')
plt.title(' 5*5 mean')
plt.axis('off')
plt.tight_layout()

plt.subplot(1, 2,2)
plt.imshow(median5_mri, cmap='gray')
plt.title('5*5 Median')
plt.axis('off')
plt.tight_layout()

plt.savefig('./55_mean_median')
plt.show()

# Part D : Custom filter

filters['custom'] = np.array([
    [1/36, 4/36, 1/36],
    [4/36, 16/36, 4/36],
    [1/36, 4/36, 1/36]
])
new_image = cross_correlation(mri, 'custom')
new_image = new_image.astype(np.uint8)

plt.subplot(1,2,1)
plt.imshow(mean_mri, cmap = 'gray')
plt.title('Mean')
plt.axis('off')
plt.tight_layout()

plt.subplot(1,2,2)
plt.imshow(new_image, cmap = 'gray')
plt.title('Custom')
plt.axis('off')
plt.tight_layout()

plt.savefig('./custom_filter')
plt.show();
