#!/usr/bin/env python
# -*- coding: utf-8 -*-

from perception import Perception

# 定义激活函数f
f = lambda x : x

class LinearUnit(Perception):
	def __init__(self,input_num):
		'''
		初始化线性单元，设置输入参数的个数
		'''
		Perception.__init__(self,input_num,f)
	
def get_train_dataset():
	'''
	构造5组数据
	'''
	# 构造训练数据
	# 输入向量列表，每一项是工作年限
	input_vectors = [[5],[3],[8],[1.4],[10.1]]
	# 期望的输出列表，月薪，注意要与输入一一对应
	labels = [5500,2300,7600,1800,11400]
	return input_vectors,labels

def train_linear_unit():
	'''
	使用数据训练线性单元
	'''
	# 创建感知器，输入参数的特征数为1（工作年限）
	lu = LinearUnit(1)
	# 训练，迭代10轮，学习速率为0.01
	input_vectors,labels = get_train_dataset()
	lu.train(input_vectors,labels,30,0.01)
	# 返回训练好的xianxingdanyuan
	return lu

def plot(linear_unit):
	import matplotlib.pyplot as plt
	input_vectors,labels = get_train_dataset()
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(map(lambda x:x[0],input_vectors),labels)
	weights = linear_unit.weights
	bias = linear_unit.bias
	x = range(0,12,1)
	y = map(lambda x:weights[0] * x + bias,x)
	ax.plot(x,y)
	plt.show()

if __name__ == '__main__':
	'''
	训练线性单元
	'''
	linear_unit = train_linear_unit()
	# 打印训练获得的权重
	print linear_unit
	plot(linear_unit)
