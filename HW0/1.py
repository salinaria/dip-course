import numpy as np

# Step 1
arr = np.random.uniform(-5000, 5000, size=80)
print('Step 1\n==============================\n',arr)

# Step 2
print('Step 2\n==============================\n',arr.dtype,type(arr))

# Step 3
arr = np.round_(arr)
print('Step 3\n==============================\n',arr,arr.dtype)

# Step 4
arr = arr.astype(np.int16)
print('Step 4\n==============================\n',arr,arr.dtype)
# Step 5
arr = np.interp(arr, (-5000, 5000), (0, 255)).astype(np.int16)
print('Step 5\n==============================\n',arr,arr.dtype)

# Step 6
arr.shape = (8,10)
print('Step 6\n==============================\n',arr)

# Step 7
arr = arr.astype(np.int8)
print('Step 7\n==============================\n',arr)