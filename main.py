import numpy as np
import math
from matplotlib import pyplot as plt
import time


def load_data(file_path, split=','):
    arr = []
    f = open(file_path)
    lines = f.readlines()
    for line in lines:
        newline = line.replace('\n', '')
        temp = newline.split(split)
        arr.append(temp)
    f.close()
    return arr


def parse_data(data, i=2):
    arr = []
    for ii in range(0, i):
        temp = []
        for item in data:
            temp.append(float(item[ii]))
        arr.append(temp)
    return arr


# 假设函数
def hypothesis(theta0, theta1, x):
    return theta0 + theta1 * x


# 计算损失值
def compute_cost(setx, sety, theta0, theta1):
    num = 0
    for i, x in enumerate(setx):
        y = sety[i]
        r = hypothesis(theta0, theta1, x)
        tmp = math.pow(r - y, 2)
        num += tmp
    m = len(setx)
    result = (1 / (2 * m)) * num
    return result


# 梯度下降
def gradient_descent(setx, sety, theta0, theta1, alpha):
    m = len(setx)
    tmp0 = 0
    tmp1 = 0
    for j in range(0, len(setx)):
        # 对 theta0 求导
        tmp0 += (1 / m) * (hypothesis(theta0, theta1, setx[j]) - sety[j])
        # tmp0 += theta0 - alpha * k / m
        # 对 theta1 求导
        tmp1 += (1 / m) * (hypothesis(theta0, theta1, setx[j]) - sety[j]) * setx[j]
    theta0_r = theta0 - (alpha * tmp0)
    theta1_r = theta1 - (alpha * tmp1)
    return theta0_r, theta1_r


# 梯度下降循环
def gradient_descent_multi(setx, sety, alpha):
    loop = True
    theta = [0, 0]
    last_j = compute_cost(setx, sety, theta[0], theta[1])
    print('[ROUND/%05d] theta0: %.6f, theta1: %.6f, result: %.6f' % (0, theta[0], theta[1], last_j))
    i = 1
    while (loop):
        theta = gradient_descent(setx, sety, theta[0], theta[1], alpha)
        j = compute_cost(setx, sety, theta[0], theta[1])
        loop = (last_j - j) > 0.0001
        last_j = j
        print('[ROUND/%05d] theta0: %.6f, theta1: %.6f, result: %.6f' % (i, theta[0], theta[1], j))
        i += 1
        # time.sleep(0.5)
    return theta


if __name__ == '__main__':
    r = load_data('./data/ex1data1.txt')
    xy = parse_data(r, i=2)
    x = xy[0]
    y = xy[1]
    r = gradient_descent_multi(x, y, 0.01)
    nx = np.linspace(4, 30, 10000)
    ny = hypothesis(r[0], r[1], nx)
    plt.title("City population vs Profit")
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.plot(x, y, "ob")
    plt.plot(nx, ny, '.r')
    plt.show()
