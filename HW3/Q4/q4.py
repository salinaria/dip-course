import cv2
import matplotlib.pyplot as plt
import numpy as np

noisy_spine = cv2.imread('./Noisy_Spine.png', cv2.IMREAD_GRAYSCALE)

# Part A : Median 5*5 filter

median_spine = cv2.medianBlur(noisy_spine, 5)

plt.subplot(1,2,1)
plt.imshow(noisy_spine, cmap='gray')
plt.title('Original')
plt.axis('off')
plt.tight_layout()

plt.subplot(1,2,2)
plt.imshow(median_spine, cmap='gray')
plt.title('Median')
plt.axis('off')
plt.tight_layout()

plt.savefig('./median')
plt.show()

# Part B : Mean 3*3 filter

avg_spine = cv2.blur(median_spine, (3, 3))

plt.subplot(1,2,1)
plt.imshow(noisy_spine, cmap='gray')
plt.title('Original')
plt.axis('off')
plt.tight_layout()

plt.subplot(1,2,2)
plt.imshow(avg_spine, cmap='gray')
plt.title('Average')
plt.axis('off')
plt.tight_layout()

plt.savefig('./mean')
plt.show()

# Part C : Laplacian filter on mean filter

laplacian_45 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
laplacian_mask = cv2.filter2D(avg_spine, ddepth=cv2.CV_64F, kernel=laplacian_45)

k = -2
laplacian = avg_spine + k * laplacian_mask

plt.subplot(1,2,1)
plt.imshow(noisy_spine, cmap='gray', vmin= 0, vmax=255)
plt.title('Original')
plt.axis('off')
plt.tight_layout()

plt.subplot(1,2,2)
plt.imshow(laplacian, cmap='gray', vmin= 0, vmax=255)
plt.title('Laplacian')
plt.axis('off')
plt.tight_layout()

plt.savefig('./laplacian')
plt.show()

# Part D : Plot above filters

plt.subplot(2, 2,1)
plt.imshow(noisy_spine, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(2, 2,2)
plt.imshow(median_spine, cmap='gray')
plt.title('Median')
plt.axis('off')

plt.subplot(2, 2,3)
plt.imshow(avg_spine, cmap='gray')
plt.title('Average')
plt.axis('off')

plt.subplot(2, 2,4)
plt.imshow(laplacian, cmap='gray', vmin= 0, vmax=255)
plt.title('Laplacian')
plt.axis('off')

plt.savefig('./d_res')
plt.show()

# Part E : Apply all filters toghter

s_avg_spine = cv2.blur(noisy_spine, (3, 3))
s_median_spine = cv2.medianBlur(s_avg_spine, 5)

s_laplacian_mask = cv2.filter2D(s_median_spine, ddepth=cv2.CV_64F, kernel=laplacian_45)
k = -2
s_result = s_median_spine + k * s_laplacian_mask

plt.imshow(s_result, cmap='gray', vmin = 0 , vmax = 255)
plt.axis('off')
plt.title('All filters')
plt.savefig('./e_res')
plt.show()
