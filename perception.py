#!/usr/bin/env python
#-*- coding: UTF-8 -*-

class Perception(object):
	def __init__(self,input_num,activator):
		'''
		初始化感知器，设置输入参数的个数，以及激活函数
		激活函数的类型为double->double
		'''
		self.activator = activator
		# 权重向量初始化为0
		self.weights = [0.0 for _ in range(input_num)]
		# 偏置项初始化为0
		self.bias = 0.0
	
	def __str__(self):
		'''
		打印学习到的权重，偏置项
		'''
		return 'weights\t:%s\nbias\t:%f\n' % (self.weights,self.bias)
	
	def predict(self,input_vector):
		'''
		输入向量，输出感知器的计算结果
		'''
		# 把input_vector[x1,x2,x3,...] 和weights[w1,w2,w3,...]打包在一起
		# 变成[(x1,w1),(x2,w2),(x3,w3),...]
		# 然后利用map函数计算[x1 * w1 ,x2 * w2,x3 * w3,...]
		# 最后利用reduce求和
		return self.activator(reduce(lambda a,b : a + b,map(lambda(x,w):x * w,zip(input_vector,self.weights)),0.0) + self.bias)
	def _update_weights(self,input_vector,output,label,rate):
		'''
		按照感知器规则更新权重
		'''
		# 把input_vector[x1,x2,x3,...]和weights[w1,w2,w3,...]打包在一起
		# 变成[(x1,w1),(x2,w2),(x3,w3),...]
		# 然后按照感知器规则更新权重 
		delta = label - output
		self.weights = map(lambda(x,w) : w + rate * delta * x,zip(input_vector,self.weights))
		self.bias += rate * delta

	def _one_iteration(self,input_vectors,labels,rate):
		'''
		一次迭代，把所有的训练数据过一遍
		'''
		# 把输入和label打包在一起，成为样本的训练列表[(input_vector,label),...]
		samples = zip(input_vectors,labels)
		# 对每个样本，按照感知器规则更新权重
		for (input_vector,label) in samples:
			# 计算感知器在当前权重下的输出
			output = self.predict(input_vector)
			# 更新权重
			self._update_weights(input_vector,output,label,rate)
	
	def train(self,input_vectors,labels,iteration,rate):
		'''
		输入训练数据：一组向量以及每个向量对应的label;训练轮数以及学习率
		'''
		for i in range(iteration):
			self._one_iteration(input_vectors,labels,rate)

def f(x):
	'''
	定义激活函数f
	'''
	return 1 if x > 0 else 0

def get_train_dataset():
	'''
	基于and真值表构造训练数据
	'''
	input_vectors = [[1,1],[1,0],[0,1],[0,0]]
	labels = [1,0,0,0]
	return input_vectors,labels

def train_and_perception():
	'''
	使用and真值表训练数据
	'''
	# 创建感知器，输入参数个数为2，激活函数为f
	p = Perception(2,f)
	# 训练，迭代10轮，学习速率为0.1
	input_vectors,labels = get_train_dataset()
	p.train(input_vectors,labels,10,0.1)
	# 返回训练好的感知器
	return p

if __name__ == '__main__':
	# 训练and感知器
	and_perception = train_and_perception()
	# 打印训练获得的权重
	print and_perception
	print '1 and 0 = %d' % and_perception.predict([1,0])
	
