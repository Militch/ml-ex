data = [[col for col in range(4)] for row in range(4)]

for row_index, row in enumerate(data):
    for col_index in range(row_index, len(row)):
        tmp = data[col_index][row_index]  # 设置一个临时变量
        data[col_index][row_index] = row[col_index]
        data[row_index][col_index] = tmp
    print('')  # 防止打印结果看上去混乱，输入一个空内容
    for r in data:  # 分步骤打印出转换结果
        print(r)