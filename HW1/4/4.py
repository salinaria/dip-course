import matplotlib.pyplot as plt
import numpy as np

background_img = plt.imread('./4/background-LE.bmp')
fullscale_img = plt.imread('./4/fullscale-LE.bmp')
object_img = plt.imread('./4/object-LE.bmp')

back_avr = []
full_avr = []
for i in range(320):
    back_avr.append(np.mean(background_img[i, :, 0]))
    full_avr.append(np.mean(fullscale_img[i, :, 0]))

new_object_img = np.zeros((320,413))
for i in range(320):
    for j in range(413):
        new_object_img[i][j] = 255 * \
            (object_img[i][j][0] - back_avr[i])/full_avr[i] - back_avr[i]

plt.axis('off')
plt.imshow(new_object_img, cmap='gray')
plt.savefig('./4/normalized.jpg', bbox_inches='tight', pad_inches=0)
plt.show()

fig, axs = plt.subplots(ncols = 2)
axs[0].imshow(object_img, cmap='gray')
axs[1].imshow(new_object_img, cmap='gray')
fig.savefig('./4/compare.jpg')
plt.show()
