from pydicom import dcmread
import matplotlib.pyplot as plt
import cv2

ds = dcmread('./1/file1.dcm')
print(ds)
plt.imshow(ds.pixel_array, cmap='gray')
#plt.show()

sec_pixel_array = ds.pixel_array[::4]
plt.imshow(sec_pixel_array, cmap='gray')
plt.savefig('./1/aforth-x.png')
#plt.show()

sec_pixel_array = sec_pixel_array[:,::4]
plt.imshow(sec_pixel_array, cmap='gray')
plt.savefig('./1/aforth-y.png')
#plt.show()



third_pixel_array = ds.pixel_array[::2][:,::2]
plt.imshow(third_pixel_array, cmap='gray')
plt.savefig('./1/halfhalf.png')
#plt.show()

near_pixel_array = cv2.resize(third_pixel_array, (512, 512), 0, 0, interpolation = cv2.INTER_NEAREST)
near_pixel_array = near_pixel_array.astype('uint8')
plt.imshow(near_pixel_array, cmap='gray')
plt.title('nearest')
#plt.show()

linear_pixel_array = cv2.resize(third_pixel_array, (512, 512), 0, 0, interpolation = cv2.INTER_LINEAR)
linear_pixel_array = linear_pixel_array.astype('uint8')
plt.imshow(linear_pixel_array, cmap='gray')  
plt.title('linear')
#plt.show()

cubic_pixel_array = cv2.resize(third_pixel_array, (512, 512), 0, 0, interpolation = cv2.INTER_CUBIC)
cubic_pixel_array = cubic_pixel_array.astype('uint8')
plt.imshow(cubic_pixel_array, cmap='gray')
plt.title('cubic')
#plt.show()

cv2.imwrite('./1/cubic_pixel_array.tif', cubic_pixel_array)
cv2.imwrite('./1/linear_pixel_array.tif', linear_pixel_array)
cv2.imwrite('./1/near_pixel_array.tif', near_pixel_array)

cv2.imwrite('./1/cubic_pixel_array.bmp', cubic_pixel_array)
cv2.imwrite('./1/linear_pixel_array.bmp', linear_pixel_array)
cv2.imwrite('./1/near_pixel_array.bmp', near_pixel_array)
