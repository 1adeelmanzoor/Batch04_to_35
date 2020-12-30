# https://numpy.org/doc/stable/user/quickstart.html
# https://www.machinelearningplus.com/python/101-numpy-exercises-python/


# implentation of any 50 numpy methods or attribute codes in jupyter notebook

# task 1
import numpy as np
np.arange(1,13).reshape((6,2))

# task 2
np.arange(1,13).reshape((6,2))

# task 3
np.arange(10,37,dtype=np.float64).reshape((3,3,3))

# task 4
arr = np.arange(9).reshape(3,3)
arr[:,[0,1]]=arr[:,[1,0]]
arr

# task 5
np.zeros((20),dtype=int).reshape(4,5)

# task 6
x = np.zeros(10)
x[[4, 7]] = 10,20
x

# task 7
x = np.arange(4, dtype=np.int64)
x[:]=0
x

# task 8
np.ones((2,5), dtype=np.uint)*6

# task 9
np.arange(2,102, 2)

# task 10
arr = np.array([[3,3,3],[4,4,4],[5,5,5]])
brr = np.array([1,2,3])
subt = arr-brr[:,None]
subt

# task 11
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
np.where(arr%2!=0,-1,arr)

# task 12
a = np.array([1,2,3])
np.r_[np.repeat(a, 3), np.tile(a, 3)]

# task 13
arr = np.array([2, 6, 1, 9, 10, 3, 27])
arr[(arr>5)&(arr<10)]

# task 14
arr = np.arange(10, 34, 1).reshape(8,3)
np.array_split(arr,4)

# task 15
arr = np.array([[ 8,  2, -2],[-4,  1,  7],[ 6,  3,  9]])
arr = arr[[1,0,2],:]
arr

# task 16
x = np.array([[1], [2], [3]])
y = np.array([[2], [3], [4]])
np.dstack((x,y))

# task 17
arr = np.arange(1,10*10+1).reshape((10,10))
np.where([(arr %3==0 ) & (arr %5==0)] ,'YES','NO')

# task 18
piaic = np.arange(100)
students = np.array([5,20,50,200,301,7001])
len(np.intersect1d(piaic , students))

# task 19
x= np.arange(1,26).reshape(5,5)
w=x.copy().transpose()
b=5
(x*w)+b

# task 20

arr = np.array([1.1, 2.1, 3.1])

arr.astype('i')

# task 21
arr = np.array([1, 2, 3, 4, 5])
arr.view()

# task 22
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
np.concatenate((arr1, arr2))

# task 23
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
np.stack((arr1, arr2), axis=1)

# task 24
np.hstack((arr1, arr2))

# task 25
np.vstack((arr1, arr2))

# task 26
arr = np.array([1, 2, 3, 4, 5, 4, 4])
np.where(arr == 4)

# task 27
arr = np.array([6, 7, 8, 9])
np.searchsorted(arr, 7)

# task 28
arr = np.array(['banana', 'cherry', 'apple'])
np.sort(arr)

# task 29
 random.randint(100)
 
# task 30
random.rand(3,4)

# task 31
random.choice([3, 5, 7, 9])

# task 32
x = [1, 2, 3, 4]
y = [4, 5, 6, 7]
z = np.add(x, y)

# task 33
def myadd(x, y):
  return x+y

myadd = np.frompyfunc(myadd, 2, 1)

print(myadd([1, 2, 3, 4], [5, 6, 7, 8]))

# task 34
arr1 = np.array([10, 20, 30, 40, 50, 60])
arr2 = np.array([20, 21, 22, 23, 24, 25])

np.subtract(arr1, arr2)


# task 35
np.multiply(arr1, arr2)

# task 36
np.divide(arr1, arr2)

# task 37
np.power(arr1, arr2)

# task 38
np.remainder(arr1, arr2)

# task 39
newarr = np.divmod(arr1, arr2)

# task 40
np.absolute(arr)

# task 41
arr = np.array([1, 1, 1, 2, 3, 4, 5, 5, 6, 7])
np.unique(arr)

# task 42
newarr = np.union1d(arr1, arr2)

# task 43
newarr = np.setdiff1d(set1, set2, assume_unique=True)

# task 44
np.sin(np.pi/2)

# task 45

arr = np.array([90, 180, 270, 360])
np.deg2rad(arr)

# task 46
np.rad2deg(arr)


# task 47
np.arcsin(1.0)

# task 48
np.prod([arr1, arr2])

# task 49
np.cumprod(arr)

# task 50
np.diff(arr, n=2)



