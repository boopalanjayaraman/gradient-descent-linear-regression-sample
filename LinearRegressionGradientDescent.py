import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import logging.config
import logging.handlers

'''
This python program is to do a sample gradient descent run on a single variable linear regression problem.
This attempts the approach to minimize the actual cost function (J) while calculating gradients of m (also known as theta / slope) and b (also known as c - constant in standard equations)
J = (1/2*n) * SUM (hi - yi)^2 where hi can be defined by m*xi + b
'''

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('LinearRegressionFirst')

logStepGradients = False

'''
run() - entry method
'''
def run():
	
	logger.info('entered run.')
	
	data_file_path = 'data.csv'
	learning_rate = 0.0001
	n = 0
	iterations = 1000
	initial_b = 0
	initial_m = 0
	
	logger.info('initialized values')
	
	#fetch values from data file
	points = np.genfromtxt(data_file_path, delimiter=',')
	
	#normalize features
	
	logger.info('fetched from datafile and generated points.')
	row_count = len(points[0:, 0:])
	logger.info('rows count: {0}'.format(row_count))
	
	#compute initial error
	initial_error = compute_error_for_points(initial_b, initial_m, points)
	logger.info('parameters value before grad descent: b = {0}, m = {1}, initial_error = {2}'.format(initial_b, initial_m, initial_error))
	
	#execute gradient descent process
	[final_b, final_m] = execute_gradient_descent(points, learning_rate, iterations, initial_b, initial_m, initial_error)
	
	#compute final error
	final_error = compute_error_for_points(final_b, final_m, points)
	logger.info('parameters value after grad descent: b = {0}, m = {1}, final_error = {2}'.format(final_b, final_m, final_error))
	
'''
execute_gradient_descent() - entry method for gradient descent 
'''
def execute_gradient_descent(points, learning_rate, iterations, initial_b, initial_m, initial_error):
	
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
		[b,m] = step_gradient(b, m, points_array, learning_rate, i)
		
		#compute the error for this iteration
		iter_error = compute_error_for_points(b, m, points)
		
		#compare and break if error increases, else continue
		logger.info('error for this iteration: {0}.'.format(iter_error))
		if(iter_error < prev_iter_error):
			logger.info('error reduced. Can continue the iterations')
		else:
			logger.info('error increased. May be crossing the minima.')
			[b,m] = [prev_b, prev_m]
			break
		
	return [b,m]

'''
step_gradient - method that executes a step of gradient
b - current b
m - current m
points_array - points array
learning_rate - learning rate that will be used to produce new b and m values 
'''
def step_gradient(b, m, points_array, learning_rate, iteration):
	
	if(logStepGradients):
		logger.info('entered step_gradient.')
		logger.info('current values: b={0}, m={1}, iteration:{2}.'.format(b, m, iteration))
	
	b_gradient = 0
	m_gradient = 0
	n = len(points_array[0:,0:])
	N = float(n)
	
	for i in range(0, n):
		xi = points_array[i, 0]
		yi = points_array[i, 1]
		#compute sum of partial derivatives of b and m (gradients)
		b_gradient += (2/N) * (yi - ((m*xi) + b)) * (-1)
		m_gradient += (2/N) * (yi - ((m*xi) + b)) * (-xi)
	
	#update b and m values using learning rate
	new_b = b - (learning_rate * b_gradient)
	new_m = m - (learning_rate * m_gradient)
	
	return [new_b, new_m]

'''
compute_error_for_points - computes error with given b and m values for all points
'''
def compute_error_for_points(b, m, points):
	
	logger.info('entered compute_error_for_points')
	logger.info('current values: b={0}, m={1}.'.format(b, m))
	
	total_error = 0
	
	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		total_error += (y - (m*x + b)) **2

	return total_error/float(len(points))
		
if __name__ == '__main__':
	run()