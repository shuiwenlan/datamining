import numpy as np

f = open('iris.txt')
Data = [
    [float(i) for i in a.split(',')[:-1]] for a in f.readlines()]
matrix = np.array(Data)#矩阵
n = matrix.shape[0]#行
d = matrix.shape[1]#列

#print(matrix)
sum1 = 0.0
hqk_matrix = np.zeros([n, n])#齐次二次核矩阵初始化，0
for i in range(n):
    for j in range(n):
        hqk_matrix[i][j] = (np.dot(matrix[i], matrix[j]))**2#齐次二次，c=0,q=2

print(hqk_matrix)
I = np.eye(n)-np.full((n, n), 1/n)
centeredK = np.dot(np.dot(I, hqk_matrix), I)#中心化
W = np.diag(np.diag(centeredK)**(-0.5))
normalizedK = np.dot(np.dot(W, centeredK), W)#标准化
######### Q1完成
######### Q2
Data2 = []
for i in range(n):   #转换到特征空间
    tran = [Data[i][0]**2, Data[i][1]**2, Data[i][2]**2, Data[i][3]**2,
             Data[i][0] * Data[i][1] * (2**0.5), Data[i][0] * Data[i][2] * (2**0.5),
             Data[i][0] * Data[i][3] * (2**0.5), Data[i][1] * Data[i][2] * (2**0.5),
             Data[i][1] * Data[i][3] * (2**0.5), Data[i][2] * Data[i][3] * (2**0.5)]
    Data2.extend(tran)
matrix2 = np.array(Data2)
mean = matrix2.mean(axis=0)#中心化
Z = matrix2 - np.ones((matrix2.shape[0], 1), dtype=float) * mean
for i in range(n):#标准化
    Z[i] = Z[i]/(np.vdot(Z[i], Z[i])**0.5)
######### Q2完成
######### Q3
hqk_matrix2 = np.zeros([n, n]) #核矩阵
for i in range(n):
    for j in range(n):
        hqk_matrix2[i][j] = (np.dot(matrix2[i], matrix2[j]))

x = 0
y = 0
while x < n:#判断是否相同
    while y < n:
        if hqk_matrix[x][y] == hqk_matrix2[x][y]:
            y = y + 1
        else:
            print("不相同")
            break
    x = x + 1
    break
    print("相同")

f.close()

