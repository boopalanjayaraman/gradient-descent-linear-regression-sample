import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import logging.config
import logging.handlers
from sklearn.datasets import load_boston

'''
This python program is to do a sample gradient descent run on a multiple variables (multivariate) linear regression problem.
Uses matrix calculations and matrix arithmetic operations.
This attempts the approach to minimize the actual cost function (J) while calculating gradients of m (also known as theta / slope) and b (also known as c - constant in standard equations).
J = (1/2*n) * SUM (hi - yi)^2 where (hi) can be defined by m*xi + b.
'''

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('LinearRegressionFirst')

logIterations = False
logStepGradients = False
logErrorComputations = False

'''
run() - entry method
'''
def run():
	
	logger.info('entered run.')
	
	data_file_path = 'data.csv'
	learning_rate = 0.0001
	n = 0
	iterations = 100000
	
	logger.info('initialized default values')
	
	#fetch values from data file
	points = np.genfromtxt(data_file_path, delimiter=',')
	points_array = np.array(points)

	feature_count = len(points_array[0,:]) - 1
	
	X = points_array[:, 0: feature_count] #first element in the range is index (starting) and then count [index:count]
	Y = points_array[:, feature_count] #feature_count is index here for the last column
	
	
	'''
	#If you want to run this with boston data for example, uncomment this block, and comment above block
	
	boston_data = load_boston()
	points = boston_data.data
	points_array = np.array(points)
	feature_count = len(points[0,:])
	X = boston_data['data']
	Y = boston_data['target']
	'''
	
	initial_b = 0 #should it be a vector too?
	#initial_m = np.zeros(feature_count)
	initial_m = np.random.randn(feature_count)
	
	logger.info('initialized parameters')
	
	#normalize features
	for i in range(feature_count):
		X[:,i] = (X[:,i] - np.min(X[:,i])) / np.max(X[:,i]) 
	
	
	logger.info('fetched from datafile and generated points.')
	row_count = len(points_array[0:, 0:])
	logger.info('rows count: {0}'.format(row_count))
	
	#compute initial error
	initial_error = compute_error_for_points(initial_b, initial_m, X, Y)
	logger.info('parameters value before grad descent: b = {0}, m = {1}, initial_error = {2}'.format(initial_b, initial_m, initial_error))
	
	#execute gradient descent process
	[final_b, final_m] = execute_gradient_descent(points, learning_rate, iterations, initial_b, initial_m, initial_error, feature_count, X , Y)
	
	#compute final error
	final_error = compute_error_for_points(final_b, final_m, X, Y)
	logger.info('parameters value after grad descent: b = {0}, m = {1}, final_error = {2}'.format(final_b, final_m, final_error))
	
'''
execute_gradient_descent() - entry method for gradient descent 
'''
def execute_gradient_descent(points, learning_rate, iterations, initial_b, initial_m, initial_error, feature_count, X, Y):
	
	logger.info('entered execute_gradient_descent.')
	
	b = initial_b
	m = initial_m
	iter_error = initial_error
	
	points_array = np.array(points)
	
	logger.info('initiated step gradient descent process.')
	for i in range(iterations):
		#store previous values of m, b and error for comparing with next iteration calculations, to decide if it improves
		[prev_b, prev_m] = [b,m]
		prev_iter_error = iter_error
		
		#execute step gradient process
		[b,m] = step_gradient(b, m, points_array, learning_rate, i, feature_count, X, Y)
		
		#compute the error for this iteration
		iter_error = compute_error_for_points(b, m, X, Y)
		
		#compare and break if error increases, else continue
		logIterationMessage('error for this iteration: {0}.'.format(iter_error))
		if(iter_error < prev_iter_error):
			logIterationMessage('error reduced. Can continue the iterations')
		else:
			logIterationMessage('error increased. May be crossing the minima.')
			[b,m] = [prev_b, prev_m]
			break
		
	return [b,m]

def logIterationMessage(message):
	if(logIterations):
		logger.info(message)
	
'''
step_gradient - method that executes a step of gradient
b - current b
m - current m
points_array - points array
learning_rate - learning rate that will be used to produce new b and m values 
'''
def step_gradient(b, m, points_array, learning_rate, iteration, feature_count, X, Y):
	
	if(logStepGradients):
		logger.info('entered step_gradient.')
		logger.info('current values: b={0}, m={1}, iteration:{2}.'.format(b, m, iteration))
	
	b_gradient = 0
	m_gradient = np.zeros(feature_count)#0
	n = len(points_array[0:,0:])
	N = float(n)
	
	#compute sum of partial derivatives of b and m (gradients)
	b_gradient += (2/N) * np.sum((Y - ((np.dot(X, m)) + b)) * (-1)) 
	m_gradient += (2/N) * np.sum(np.dot((Y - ((np.dot(X, m)) + b)) , (-X)))
	
	#print(m_gradient)
	
	'''
		#for reference
		#compute sum of partial derivatives of b and m (gradients)
		b_gradient += (2/N) * (yi - ((m*xi) + b)) * (-1)
		m_gradient += (2/N) * (yi - ((m*xi) + b)) * (-xi)'''
	
	#update b and m values using learning rate
	new_b = b - (learning_rate * b_gradient)
	new_m = m - (learning_rate * m_gradient)
	
	return [new_b, new_m]

'''
compute_error_for_points - computes error with given b and m values for all points
'''
def compute_error_for_points(b, m, X, Y):
	
	if(logErrorComputations):
		logger.info('entered compute_error_for_points')
		logger.info('current values: b={0}, m={1}.'.format(b, m))
	
	total_error = 0
	
	error = (Y - (np.dot(X, m) + b))**2
	
	total_error = np.sum(error)
	error_mean = np.mean(error)
	
	return error_mean
		
if __name__ == '__main__':
	run()