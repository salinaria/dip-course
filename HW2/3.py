import cv2
import matplotlib.pyplot as plt
import numpy as np

spine = cv2.imread('spine.tif', cv2.IMREAD_GRAYSCALE)
spine = spine.astype(np.float32)


def transform(img, name, gamma=1.0):
    ans = np.zeros(img.shape)
    
    if name == 'power law':    
        for i in range(512):
            for j in range(512):
                ans[i][j] = (img[i][j] ** gamma) * (254**(1-gamma))
    
    elif name == "contrast stretch":
        r1 = np.min(img)
        r2 = np.max(img)

        s1 = 0
        s2 = 255

        for i in range(512):
            for j in range(512):
                ans[i][j] = (img[i][j]-r1)*(255)/(r2-r1)
    
    return ans

plt.subplot(1, 3, 1)
plt.title("original image")
plt.imshow(spine, cmap='gray', vmin=0, vmax=255, )
plt.axis('off')

contrast_stretched_img = transform(spine, "contrast stretch")
powered_img = transform(spine, "power law", 0.5)

powered_contrasted_img = transform(contrast_stretched_img, "power law", 0.5)
contrasted_powered_img = transform(powered_img, "contrast stretch")

plt.subplot(2, 3, 2)
plt.title("contrast stretch")
plt.imshow(contrast_stretched_img, cmap='gray', vmin=0, vmax=255)
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("power")
plt.imshow(powered_img, cmap='gray', vmin=0, vmax=255)
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("power of contrast")
plt.imshow(powered_contrasted_img, cmap='gray', vmin=0, vmax=255)
plt.axis('off')

plt.subplot(2, 3, 6)
plt.title("contrast of power")
plt.imshow(contrasted_powered_img, cmap='gray', vmin=0, vmax=255)
plt.axis('off')


plt.savefig('./res_q3',cmap='gray', vmin=0, vmax=255)
plt.show()

