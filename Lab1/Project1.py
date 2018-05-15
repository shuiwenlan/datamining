import numpy as np
import matplotlib.pyplot as plt

f = open('magic04.txt')
Data = [
    [float(i) for i in a.split(',')[:-1]] for a in f.readlines()]
matrix = np.array(Data)#矩阵
n = matrix.shape[0]#行
d = matrix.shape[1]#列
#print(matrix)
mean = np.mean(matrix, axis=0)
print(mean)
######## 均值向量   Q1完成

######## Q2
Z = matrix - np.ones([n, 1]) * mean#中心化矩阵
#print(Z)
t_Z = np.transpose(Z)#转置
inner_product = np.dot(t_Z, Z)/n#内积  Q2
print(inner_product)
######## Q2完成

######## Q3
outer_product = np.zeros([d, d])
for zi in Z:

    zi = np.transpose(zi)
    t_zi = np.transpose(zi)
    outer_product += t_zi * zi
print(outer_product/n) #外积   Q3
######## Q3完成

######## Q4
attribute1 = Z[:, 0]
attribute2 = Z[:, 1]
cosine = np.dot(attribute1, attribute2)/(np.dot(attribute1, attribute1) *
                                         np.dot(attribute2, attribute2))
print(cosine)#Q4  还要画图
plt.scatter(attribute1,attribute2)
plt.show()
######## Q4完成

######## Q5
vars = np.var(matrix, axis=0)#方差        Q5中还要画图
#print(vars[0])

data = attribute1


def normfunction(x):
    pdf = np.exp(-((x - mean[0])**2)/(2*vars[0]))/(np.sqrt(2*np.pi*vars[0]))
    return pdf


x = np.linspace(mean[0] - 300, mean[0] + 300, 1000)
y = [normfunction(i) for i in x]
plt.plot(x, y)
plt.show()
######## Q5完成

######## Q6
a = np.where(vars == np.max(vars))[0][0]
b = np.where(vars == np.min(vars))[0][0]#最小方差的位置
print(a)
print(b)
######## Q6完成

######## Q7
cov = {
}
for i in range(d):
    for j in range(d):
        if i >= j:
            continue
        cov['' + str(i) + '-' + str(j)] = np.cov(Z[:, i], Z[:, j])[0][1]
max_cov = min_cov = '0-1'
for (key,value) in cov.items():
    if value > cov[max_cov]:
        max_cov=key
    if value < cov[min_cov]:
        min_cov=key
print(max_cov)
print(min_cov)

f.close()
