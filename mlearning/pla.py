# -×-coding:utf-8-*-

'''
PLA算法实现
'''


ITERATION = 70
W = [1, 1, 1]


def createData():
    lines_set = open('./data_pla.txt').readlines()
    linesTrain = lines_set[1:7]  # 测试数据
    linesTest = lines_set[9:13]  # 训练数据

    trainDataList = processData(linesTrain)  # 生成训练集（二维列表）
    testDataList = processData(linesTest)  # 生成测试集（二维列表）
    return trainDataList, testDataList


def processData(lines):  # 按行处理从txt中读到的训练集（测试集）数据
    dataList = []
    for line in lines:  # 逐行读取txt文档里的训练集
        dataLine = line.strip().split()  # 按空格切割一行训练数据（字符串）
        dataLine = [int(data) for data in dataLine]  # 字符串转int
        dataList.append(dataLine)  # 添加到训练数据列表
    return dataList


def sign(W, dataList):  # 符号函数
    sum = 0
    for i in range(len(W)):
        sum += W[i] * dataList[i]
    if sum > 0:
        return 1
    else:
        return -1


def renewW(W, trainData):  # 更新W
    signResult = sign(W, trainData)
    if signResult == trainData[-1]:
        return W
    for k in range(len(W)):
        W[k] = W[k] + trainData[-1] * trainData[k]
    return W


def trainW(W, trainDatas):  # 训练W
    newW = []
    for num in range(ITERATION):
        index = num % len(trainDatas)
        newW = renewW(W, trainDatas[index])
    return newW


def predictTestData(W, trainDatas, testDatas):  # 预测测试数据集
    W = trainW(W, trainDatas)
    print W
    for i in range(len(testDatas)):
        result = sign(W, testDatas[i])
        print result


trainDatas, testDatas = createData()

predictTestData(W, trainDatas, testDatas)
