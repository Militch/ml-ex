import numpy as np
from matplotlib import pyplot as plt


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


if __name__ == '__main__':
    r = load_data('./data/ex1data1.txt')
    xy = parse_data(r, i=3)
    x = xy[0]
    y = xy[1]
    plt.title("City population vs Profit")
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.plot(x, y, "ob",)
    plt.show()