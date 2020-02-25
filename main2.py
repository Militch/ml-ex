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


# 矩阵内积
def matrix_dot(ms_1, ms_2):
    row_1 = len(ms_1)
    column_1 = len(ms_1[0])
    column_2 = len(ms_2[0])
    out = [([0] * column_2) for i in range(row_1)]
    for i in range(row_1):
        for j in range(column_2):
            result = 0
            for k in range(column_1):
                result += ms_1[i][k] * ms_2[k][j]
            out[i][j] = result
    return out


# 矩阵转置
def matrix_t(ms):
    out = [([0] * len(ms)) for i in range(len(ms[0]))]
    for i in range(len(ms)):
        for j in range(len(ms[i])):
            out[j][i] = ms[i][j]
    return out


# 假设函数
def hypothesis(thetas, params):
    ts_m = [thetas]
    xs_m = [params]
    xs_m1 = matrix_t(xs_m)
    return matrix_dot(ts_m, xs_m1)[0][0]


# 计算损失值
def compute_cost(thetas, params, outputs):
    num = 0
    m = len(params)
    for i in range(m):
        y = outputs[i]
        v = hypothesis(thetas, params[i])
        tmp = math.pow(v - y, 2)
        num += tmp
    result = (1 / (2 * m)) * num
    return result


# 梯度下降
def gradient_descent(thetas, params, outputs, alpha):
    m = len(params)
    thetas_sign_len = len(thetas)
    tss = [0] * thetas_sign_len
    outs = [0] * thetas_sign_len
    for i in range(m):
        for j in range(len(params[i])):
            tss[j] += (1 / m) * (hypothesis(thetas, params[i]) - outputs[i]) * params[i][j]
        for j in range(thetas_sign_len):
            outs[j] = thetas[j] - (alpha * tss[j])
    return outs


# 梯度下降循环
def gradient_descent_multi(params, outputs, alpha):
    loop = True
    params_len = len(params[0])
    # thetas = [([0] * params_len) for i in range(len(params))]
    thetas = [0] * params_len
    last_j = compute_cost(thetas, params, outputs)
    print('[ROUND/%05d] t: %s, result: %.6f' % (0, thetas, last_j))
    i = 1
    while (loop):
        thetas = gradient_descent(thetas, params, outputs, alpha)
        j = compute_cost(thetas, params, outputs)
        loop = (last_j - j) > 0.0001
        last_j = j
        print('[ROUND/%05d] t: %s, result: %.6f' % (i, thetas, j))
        i += 1
        # time.sleep(0.5)
    return thetas


if __name__ == '__main__':
    r = load_data('./data/ex1data1.txt')
    xy = parse_data(r, i=2)
    xx = xy[0]
    x_count = len(xy[0])
    x = [([1] * 2) for i in range(x_count)]
    for i in range(x_count):
        o = xx[i]
        x[i][1] = o
    params_list = matrix_t([x])
    outputs_list = xy[1]
    r = gradient_descent_multi(x, outputs_list, 0.01)
    nx = np.linspace(1, 30, 1000)
    nxx = nx.tolist()
    ny = [0] * len(nxx)
    for i in range(len(ny)):
        ny[i] = hypothesis(r, [1, nxx[i]])
    plt.title("City population vs Profit [main2]")
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.plot(xy[0], xy[1], ".b")
    plt.plot(nxx, ny, 'r')
    plt.show()

