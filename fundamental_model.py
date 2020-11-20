import numpy as np
import tensorflow as tf
import time
from general import *
from OneHotEncoder import OneHotEncoder

class FundamentalModel:
    steps = 100000000
    steps = 200000
    learn_rate = 0.001
    def __init__(self, prices, earnings):
        self.prices = np.array(prices)
        self.earnings = earnings
        self.sess = tf.Session()
    
    def create_linear_layers(self, load_previous=False):
        self.x = tf.placeholder(tf.float32, [len(self.earnings)], name = "x")
        if not load_previous:
            self.W = tf.Variable(tf.zeros([1]), name = "W")
        self.predicted_prices = tf.multiply(self.x, self.W)   
        self.predicted_prices = tf.maximum(self.predicted_prices, 0.0)   
    
    def train(self, load_previous=False):     
        self.actual_prices = tf.placeholder(tf.float32, len(self.earnings))
        self.create_linear_layers(load_previous)
        self.differences = tf.abs(tf.subtract(self.actual_prices, self.predicted_prices))
        self.cost = tf.reduce_sum(self.differences)
        self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.cost)	
        feed = {self.x: self.earnings, self.actual_prices: self.prices}
        init = tf.initialize_all_variables()
        self.sess.run(init)
        for i in range(self.steps):
            print(i)
            self.sess.run(self.train_step, feed_dict = feed)
            print("After %d iterations:" % i)
            print("W: %s" % self.sess.run(self.W))
            print(self.sess.run(self.cost, feed_dict = feed))
            print(self.sess.run(self.predicted_prices, feed_dict = feed))
            print(self.sess.run(self.actual_prices, feed_dict = feed))  
            self.feed = feed

class FundamentalModel2:
    steps = 200000
    learn_rate = 0.001
    def __init__(self, prices, earnings, current_prices):
        self.prices = np.array(prices)
        self.earnings = earnings
        self.current_prices_array = current_prices
        self.sess = tf.Session()
    
    def create_linear_layers(self, load_previous=False):
        self.x = tf.placeholder(tf.float32, [len(self.earnings)], name = "x")
        self.current_prices = tf.placeholder(tf.float32, [len(self.current_prices_array)], name = "current_prices")
        if not load_previous:
            self.W = tf.Variable(tf.zeros([1]), name = "W")
            self.W2 = tf.Variable(tf.zeros([1]), name = "W2")
            self.constant_multiplier = tf.Variable(tf.ones([1]), name = "constant_multiplier")
        self.predicted_prices = tf.multiply(self.x, self.W)
        self.predicted_prices = tf.multiply(self.predicted_prices, self.constant_multiplier)
        self.predicted_prices = tf.maximum(self.predicted_prices, 0.0) 
        self.final_predicted_prices = tf.add(tf.multiply(self.current_prices, self.W2), tf.multiply(self.predicted_prices, tf.subtract(1.0, self.W2)))
   
    def train(self, load_previous=False):     
        self.actual_prices = tf.placeholder(tf.float32, len(self.earnings))
        self.create_linear_layers(load_previous)
        self.cost = tf.abs(tf.subtract(self.actual_prices, self.final_predicted_prices))
        self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.cost)	
        self.clip_op = tf.assign(self.W2, tf.clip_by_value(self.W2, 0.0, 0.99))
        feed = {self.x: self.earnings, self.actual_prices: self.prices, self.current_prices: self.current_prices_array}
        init = tf.initialize_all_variables()
        self.sess.run(init)
        for i in range(self.steps):
            print(i)
            self.sess.run(self.train_step, feed_dict = feed)
            self.sess.run(self.clip_op)
            print("After %d iterations:" % i)
            print("W: %s" % self.sess.run(self.W))
            print("W2: %s" % self.sess.run(self.W2)) 
            print("constant multiplier: %s" % self.sess.run(self.constant_multiplier))  
            print(self.sess.run(self.final_predicted_prices, feed_dict = feed))   
            print(self.sess.run(self.actual_prices, feed_dict = feed))        
            self.feed = feed

class FundamentalModel3:
    steps = 20000
    learn_rate = 0.001
    def __init__(self, prices, earnings, current_prices):
        self.prices = np.array(prices)
        self.earnings = earnings
        self.current_prices_array = current_prices
        self.sess = tf.Session()
    
    def create_linear_layers(self, load_previous=False):
        self.x = tf.placeholder(tf.float32, [len(self.earnings)], name = "x")
        self.current_prices = tf.placeholder(tf.float32, [len(self.current_prices_array)], name = "current_prices")
        if not load_previous:
            self.coefficient1 = tf.Variable(tf.zeros([1]), name = "coefficient1")
            self.coefficient2 = tf.Variable(tf.zeros([1]), name = "coefficient2")
        self.predicted_prices = tf.multiply(self.x, self.coefficient1)
        self.predicted_prices = tf.maximum(self.predicted_prices, 0.0) 
        self.final_predicted_prices = tf.add(tf.multiply(self.current_prices, self.coefficient2), self.predicted_prices)
        self.actual_prices = tf.placeholder(tf.float32, len(self.earnings))
        self.cost = tf.abs(tf.subtract(self.actual_prices, self.final_predicted_prices))
        
    def train(self, load_previous=False):  
        self.create_linear_layers(load_previous)
        self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.cost)	
        feed = {self.x: self.earnings, self.actual_prices: self.prices, self.current_prices: self.current_prices_array}
        init = tf.initialize_all_variables()
        self.sess.run(init)
        print(self.sess.run(self.coefficient1))
        self.feed = feed
        for i in range(self.steps):
            print(i)
            self.sess.run(self.train_step, feed_dict = feed)
            print("After %d iterations:" % i)
            print("coefficient1: %s" % self.sess.run(self.coefficient1))
            print("coefficient2: %s" % self.sess.run(self.coefficient2)) 

    def save_weights(self, filename):
        file = open(filename, 'w')
        weights = np.concatenate(([self.sess.run(self.coefficient1)], [self.sess.run(self.coefficient2)]))
        weights = [str(x) for x in weights]
        file.write('\t'.join(weights))
    
    def load_weights(self, filename):
        file = open(filename, 'r').read()
        weights = file.split('\t')
        self.coefficient1 = tf.Variable(tf.constant(eval(weights[0])))
        self.coefficient2 = tf.Variable(tf.constant(eval(weights[1])))

    def predict(self, prices, current_prices, earnings, load_previous = True):
        self.prices = prices
        self.current_prices_array = current_prices
        self.earnings = earnings
        self.create_linear_layers(load_previous)
        init = tf.initialize_all_variables()
        self.sess.run(init)
        self.feed = {}
        self.feed[self.x] = earnings
        self.feed[self.current_prices] = self.current_prices_array
        self.feed[self.actual_prices] = self.prices
        return self.sess.run(self.final_predicted_prices, feed_dict = self.feed)

#             print(self.sess.run(self.final_predicted_prices, feed_dict = feed))   
#             print(self.sess.run(self.actual_prices, feed_dict = feed))        
            




#         

# 
#     def produce_prediction_matrix(self, number_of_days, days_in_advance, load_previous = True, already_initialised = False):
#         prediction_matrix = self.predict(load_previous=load_previous, already_initialised=already_initialised)
#         prediction_matrix = np.ndarray.flatten(prediction_matrix)
#         print(prediction_matrix)
#         prediction_matrix = np.concatenate((rep(np.nan, number_of_days+days_in_advance), prediction_matrix))
#         return prediction_matrix
