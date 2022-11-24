import numpy as np
import matplotlib.pyplot as plt

# Constant variables
STD_NUM = 9831136
R_CIRCLE = 50

# Draw a circle
def circle_draw(r):
    arr = np.zeros((2*r+1,2*r+1))
    arr = arr.astype(np.uint8)
    for i in range(2*r+1):
        for j in range(2*r+1):
            if (i-r)**2 + (j-r)**2 <= r**2:
                arr[i,j] =255
    return arr

circ = circle_draw(R_CIRCLE)

# Add Noise to circle
def noisy(circle,noise_max):
    for i in range(circle.shape[0]):
        for j in range(circle.shape[1]):
            noise = np.random.uniform(0, noise_max, size=1)
            if circle[i,j] == 0:
                circle[i,j] += int(noise)
            else:
                circle[i,j] -= int(noise)
    return circle

# Calculate maximum available noise
std_sum = 0
std_num = STD_NUM
    
while std_num > 1:
    std_sum += std_num % 10
    std_num = std_num // 10
noise_max = 40 + std_sum % 12    

circle = noisy(circ,noise_max)

# Check type of our ndarray
print(circle.dtype)

figure = plt.figure(tight_layout=True)
(axes1, axes2) = figure.subplots(nrows=1, ncols=2)

axes1.imshow(circle, cmap = 'gray')
axes1.set_title('Noise: 0-47')

axes2.imshow(circle_draw(R_CIRCLE), cmap = 'gray')
axes2.set_title("R: {}".format(R_CIRCLE))

plt.savefig('./HW0-Image-9831136')
plt.show()