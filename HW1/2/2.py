import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
cap = cv2.VideoCapture('./2/MRI-Head.avi')

avr = 0
c = 0

first = 0
while(cap.isOpened()):
  ret, frame = cap.read()
  if first == 0:
        first_frame = frame
        first = 1
  if ret != False:
    cv2.imshow('frame',frame)
    avr += np.array(frame,dtype = 'uint16')  
    c += 1
  time.sleep(0.1)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break


cap.release()
cv2.destroyAllWindows()

avr  = avr / float(c)

fig, axs = plt.subplots(ncols = 2)
axs[0].imshow(np.array(avr,dtype='uint8'))
axs[1].imshow(first_frame)
fig.savefig('./2/compare.jpg')
plt.show()

mask1 = np.load('./2/mask1.npy')
mask2 = np.load('./2/mask2.npy')

first = np.multiply(avr[:,:,0], mask1)
second = np.multiply(avr[:,:,0], mask2)

plt.axis('off')
plt.imshow(first + second, cmap='gray')
plt.savefig('./2/masked.jpg')
plt.show()