import numpy as np

array = np.array([[1, 5],   
                  [9, 0],
                  [4, 8]])

array_3d = np.array([
    [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12]],
    
    [[13, 14, 15, 16],
     [17, 18, 19, 20],
     [21, 22, 23, 24]]
])
 
array2 = np.array([1,2,3])
array3 = array2 + 1
array4 = np.exp(array2)
array5 = array4 + 1
# print(array4/array5)

y = np.zeros(array2.shape[0])
# print(y.shape)

product = np.matmul(array2, array)
# print(product.shape)

# print(np.matmul(np.transpose(array), array))
hist = np.array([1, 2, 3, 4, 5])
print(hist[1:]) 
print(hist[:-1])
state_data = np.log(hist[1:] / hist[:-1])
print(state_data)
state_data = np.pad(state_data, (120 - len(state_data), 0), 'constant')
print(state_data)
# print(np.array([])/np.array([]))