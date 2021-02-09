import math
import pandas as pd 
import numpy as np
import random


def get_graph_for_day(day, max_neighbors):
	file1 = open('', 'r')
	lines = file1.readlines()
	adjacency_matrix = []
	for line in lines:
		line = line.split(',')[:-1]
		temp = []
		try:
			for index in random.sample(range(1, len(line)), max_neighbors):
				temp.append(int(line[int(index)]))
		except ValueError:
			for index in range(1, len(line)):
				temp.append(int(line[int(index)]))
		adjacency_matrix.append(temp)

	relation_e = []
	for index, row in enumerate(adjacency_matrix):
		temp = []
		if index in range(0, 82831):
			for element in row:
				if element in range(0, 82831):
					temp.append(0)
				elif element in range(82831, 82831 + 74855):
					temp.append(5)
				elif element in range(82831 + 74855, 82831 + 74855 + 30139):
					temp.append(6)
				elif element in range(82831 + 74855 + 30139, 82831 + 74855 + 30139 + 55):
					temp.append(7)
				elif element in range(82831 + 74855 + 3013 + 55, 82831 + 74855 + 30139 + 55 + 89493):
					temp.append(8)
		elif index in range(82831, 82831 + 74855):
			for element in row:
				if element in range(0, 82831):
					temp.append(5)
				elif element in range(82831, 82831 + 74855):
					temp.append(1)
				elif element in range(82831 + 74855, 82831 + 74855 + 30139):
					temp.append(9)
				elif element in range(82831 + 74855 + 30139, 82831 + 74855 + 30139 + 55):
					temp.append(10)
				elif element in range(82831 + 74855 + 3013 + 55, 82831 + 74855 + 30139 + 55 + 89493):
					temp.append(11)
		elif index in range(82831 + 74855, 82831 + 74855 + 30139):
			for element in row:
				if element in range(0, 82831):
					temp.append(6)
				elif element in range(82831, 82831 + 74855):
					temp.append(9)
				elif element in range(82831 + 74855, 82831 + 74855 + 30139):
					temp.append(2)
				elif element in range(82831 + 74855 + 30139, 82831 + 74855 + 30139 + 55):
					temp.append(12)
				elif element in range(82831 + 74855 + 3013 + 55, 82831 + 74855 + 30139 + 55 + 89493):
					temp.append(13)
		elif index in range(82831 + 74855 + 30139, 82831 + 74855 + 30139 + 55):
			for element in row:
				if element in range(0, 82831):
					temp.append(7)
				elif element in range(82831, 82831 + 74855):
					temp.append(10)
				elif element in range(82831 + 74855, 82831 + 74855 + 30139):
					temp.append(12)
				elif element in range(82831 + 74855 + 30139, 82831 + 74855 + 30139 + 55):
					temp.append(3)
				elif element in range(82831 + 74855 + 3013 + 55, 82831 + 74855 + 30139 + 55 + 89493):
					temp.append(14)
		elif index in range(82831 + 74855 + 3013 + 55, 82831 + 74855 + 30139 + 55 + 89493):
			for element in row:
				if element in range(0, 82831):
					temp.append(8)
				elif element in range(82831, 82831 + 74855):
					temp.append(11)
				elif element in range(82831 + 74855, 82831 + 74855 + 30139):
					temp.append(13)
				elif element in range(82831 + 74855 + 30139, 82831 + 74855 + 30139 + 55):
					temp.append(14)
				elif element in range(82831 + 74855 + 3013 + 55, 82831 + 74855 + 30139 + 55 + 89493):
					temp.append(4)
		relation_e.append(temp)

	point_e = [0] * 82831 + [1] * 74855 + [2] * 30139 + [3] * 55 + [4] * 89493
	entity_index = [[x for x in range(0, 82831)]] + \
					[[x for x in range(82831, 82831 + 74855)]] + \
					[[x for x in range(82831 + 74855, 82831 + 74855 + 30139)]] + \
					[[x for x in range(82831 + 74855 + 30139, 82831 + 74855 + 30139 + 55)]] + \
					[[x for x in range(82831 + 74855 + 30139 + 55, 82831 + 74855 + 30139 + 55 + 89493)]]

	for index in range(len(adjacency_matrix)):
		temp = [x + 1 for x in adjacency_matrix[index]]
		if len(temp) < max_neighbors:
			temp += [0] * (max_neighbors - len(temp))
		adjacency_matrix[index] = temp

	for index in range(len(relation_e)):
		temp = relation_e[index]
		if len(temp) < max_neighbors:
			temp += [0] * (max_neighbors - len(temp))
		relation_e[index] = temp

	df = pd.read_csv('' + str(day) + '.csv', header = None)
	feat = df.values[:, :-1]
	return feat, adjacency_matrix, point_e, relation_e, entity_index

def get_label():
	label_raw = pd.read_csv('', header = None)[1].values
	train_mask = []
	test_mask = []
	train_label = []
	test_label = []
	for index, element in enumerate(label_raw):
		if abs(element)>1:
			test_mask.append(index)
			if element > 0:
				test_label.append(1)
			else:
				test_label.append(0)
		else:
			train_mask.append(index)
			if element > 0:
				train_label.append(1)
			else:
				train_label.append(0)

	train_mask = np.array(train_mask)
	test_mask = np.array(test_mask)
	train_label = np.array(train_label)
	test_label = np.array(test_label)

	return train_mask, test_mask, train_label, test_label
