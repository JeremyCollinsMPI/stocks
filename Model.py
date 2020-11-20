import numpy as np
import tensorflow as tf
import time
from general import *
from OneHotEncoder import OneHotEncoder



class NewModel_latest:

    def __init__(self, actual_buy_prices, actual_sell_prices, last_sell_prices, dependent_variables, positive = True, include_intercept = False, load_previous=False, weights=None, bias=None):
        

        self.steps = 5000
        self.learn_rate = 0.001
        self.dependent_variables = np.array(dependent_variables)
        self.actual_buy_prices_array = np.array(actual_buy_prices)
        self.actual_sell_prices_array = np.array(actual_sell_prices)
        self.last_sell_prices_array = last_sell_prices
        self.positive = positive
        self.include_intercept = include_intercept
        self.one_hot_encodings = []	
        self.other_vectors = []
        self.sess = tf.Session()
        self.x = tf.placeholder(tf.float32, [None, len(self.dependent_variables[0])], name = "x")
        x = rep(0.0, len(self.dependent_variables[0]))
        x[-1] = 1.0
        x = np.array(x, dtype = np.float32)
        x = np.reshape(x, (len(x), 1))
        if load_previous:
          self.W_buy = tf.convert_to_tensor(weights, tf.float32)
          self.bias = tf.convert_to_tensor(bias, tf.float32)
        else:
          self.W_buy = tf.Variable(tf.constant(x), name = "W_buy")
          self.bias = tf.Variable(tf.constant(np.array([0.0]), dtype = np.float32))
        self.predicted_sell_prices = tf.matmul(self.x, self.W_buy) * self.bias
        self.actual_buy_prices = tf.placeholder(tf.float32, [None, 1])
        self.actual_sell_prices = tf.placeholder(tf.float32, [None, 1])
        self.loss = tf.abs(self.actual_sell_prices - self.predicted_sell_prices)
        self.nonabsolute_loss = tf.reduce_mean(self.actual_sell_prices - self.predicted_sell_prices)
        self.total_loss = tf.reduce_sum(self.loss)
        fee = 0.0003
        threshold = 1.0 + fee
        self.will_buy = tf.cast(tf.less(self.actual_buy_prices * threshold, self.predicted_sell_prices), tf.float32)        
        self.will_not_buy = 1.0 - self.will_buy
        self.hypothetical_returns = self.actual_sell_prices / self.actual_buy_prices
        self.hypothetical_returns = self.hypothetical_returns - fee
        self.returns = self.will_buy * self.hypothetical_returns
        self.average_return = tf.reduce_sum(self.returns) / tf.reduce_sum(self.will_buy) 
#         self.total_return = tf.reduce_prod(self.returns + self.will_not_buy)

        self.total_return = tf.reduce_sum(self.returns + self.will_not_buy - 1)

        
#     def create_convolutional_network(self):
#         self.cnn_input_layer = tf.placeholder(tf.float32, [len(self.dependent_variables), len(self.dependent_variables[0])])
#         self.cnn_input_layer_reshaped = tf.reshape(self.cnn_input_layer, [-1, len(self.dependent_variables[0]), 1])
#         self.cnn_conv1_filters = 5
#         self.cnn_conv1 = tf.layers.conv1d(self.cnn_input_layer_reshaped, filters = self.cnn_conv1_filters, kernel_size = 2, padding = "same", activation = tf.nn.relu)
#         self.cnn_last = self.cnn_conv1
#         number_of_cnn_outputs = 1
#         cnn_last_shape = tf.shape(self.cnn_last)
#         cnn_output_length = len(self.dependent_variables[0]) * self.cnn_conv1_filters   
#         self.cnn_last_flat = tf.reshape(self.cnn_last, [-1, cnn_output_length])
#         self.cnn_output = tf.layers.dense(self.cnn_last_flat, units = number_of_cnn_outputs, activation = tf.nn.sigmoid)
#         self.cnn_coefficients =  tf.Variable(tf.zeros([number_of_cnn_outputs, 1]))
#         self.cnn_to_add = tf.matmul(self.cnn_output, self.cnn_coefficients)
#         self.cnn_to_add = self.cnn_to_add - tf.reduce_mean(self.cnn_to_add)
#         self.y = self.y + self.cnn_to_add 

    def train(self):     
        self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.total_loss)	
        self.clip_op = tf.assign(self.W_buy, tf.clip_by_value(self.W_buy, 0, 10000))
        self.weights_total = tf.reduce_sum(self.W_buy)
        self.normalise_step = tf.assign(self.W_buy, self.W_buy / self.weights_total)
        '''
        find total of weights
        '''
        feed = {self.x: self.dependent_variables, self.actual_buy_prices: self.actual_buy_prices_array, self.actual_sell_prices: self.actual_sell_prices_array}        
        init = tf.initialize_all_variables()
        self.sess.run(init)
        '''
        want to factor in trading fee
        
        so you want average return; assume say that you trade $1000 each time.
        then the threshold needs to be 1.007
        you have the return, e.g. 0.97.
        you then need to multiply this by 1000 and subtract 7.
        or simply subtract 0.007
        
        
        '''

        for i in range(self.steps):
            print(i)
            self.sess.run(self.train_step, feed_dict = feed) 
            self.sess.run(self.clip_op)
            print(self.sess.run(self.W_buy))
#             self.sess.run(self.normalise_step, feed_dict = feed)      
            print("After %d iterations:" % i)
            total_loss = self.sess.run(self.total_loss, feed_dict=feed)
            print(total_loss)
            print(self.sess.run(self.total_return, feed_dict=feed))


    def test(self, test_actual_buy_prices, test_actual_sell_prices, test_last_sell_prices, test_dependent_variables):
        feed = {self.x: test_dependent_variables, self.actual_buy_prices: test_actual_buy_prices, self.actual_sell_prices: test_actual_sell_prices}                
        print(self.sess.run(self.average_return, feed_dict=feed))
        print(self.sess.run(tf.reduce_sum(self.will_buy), feed_dict=feed))
        print(self.sess.run(self.total_return, feed_dict=feed))
#         print(self.sess.run(self.nonabsolute_loss, feed_dict=feed))
        return self.sess.run(self.predicted_sell_prices, feed_dict=feed), self.sess.run(self.returns, feed_dict=feed)
#         proposed_buy_prices = self.sess.run(self.proposed_buy_prices, feed_dict=feed)
#         actual_buy_prices = self.sess.run(self.actual_buy_prices, feed_dict=feed)
#         print(proposed_buy_prices[np.where(proposed_buy_prices > actual_buy_prices)])
#         print(actual_buy_prices[np.where(proposed_buy_prices > actual_buy_prices)])
#         print(proposed_buy_prices)
#         print(actual_buy_prices)


class NewModel_6:

    def __init__(self, actual_buy_prices, actual_sell_prices, last_sell_prices, dependent_variables, positive = True, include_intercept = False, load_previous=False, weights=None):
        

        self.steps = 5000
        self.learn_rate = 0.001
        self.dependent_variables = np.array(dependent_variables)
        self.actual_buy_prices_array = np.array(actual_buy_prices)
        self.actual_sell_prices_array = np.array(actual_sell_prices)
        self.last_sell_prices_array = last_sell_prices
        self.positive = positive
        self.include_intercept = include_intercept
        self.one_hot_encodings = []	
        self.other_vectors = []
        self.sess = tf.Session()
        self.x = tf.placeholder(tf.float32, [None, len(self.dependent_variables[0])], name = "x")
        x = rep(0.0, len(self.dependent_variables[0]))
        x[-1] = 1.0
        x = np.array(x, dtype = np.float32)
        x = np.reshape(x, (len(x), 1))
        if load_previous:
          self.W_buy = tf.convert_to_tensor(weights, tf.float32)
        else:
          self.W_buy = tf.Variable(tf.constant(x), name = "W_buy")
        self.predicted_sell_prices = tf.matmul(self.x, self.W_buy)
        self.actual_buy_prices = tf.placeholder(tf.float32, [None, 1])
        self.actual_sell_prices = tf.placeholder(tf.float32, [None, 1])
        self.loss = tf.abs(self.actual_sell_prices - self.predicted_sell_prices)
        self.total_loss = tf.reduce_sum(self.loss)
        fee = 0.0007
        threshold = 1.0 + fee
        self.will_buy = tf.cast(tf.less(self.actual_buy_prices * threshold, self.predicted_sell_prices), tf.float32)
        self.will_not_buy = 1.0 - self.will_buy
        self.hypothetical_returns = self.actual_sell_prices / self.actual_buy_prices
        self.returns = self.will_buy * self.hypothetical_returns
        self.average_return = tf.reduce_sum(self.returns) / tf.reduce_sum(self.will_buy) 
        self.average_return = self.average_return - fee    
        self.total_return = tf.reduce_prod(self.returns + self.will_not_buy)

        
        '''
        now making just accuracy.      
        return is defined by having a predicted sell price; if actual buy price is lower than that, buy it.
        then use actual_sell_price / actual_buy_price as the return.
        '''

    def train(self):     
        self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.total_loss)	
        
        self.weights_total = tf.reduce_sum(self.W_buy)
        self.normalise_step = tf.assign(self.W_buy, self.W_buy / self.weights_total)
        '''
        find total of weights
        '''
        feed = {self.x: self.dependent_variables, self.actual_buy_prices: self.actual_buy_prices_array, self.actual_sell_prices: self.actual_sell_prices_array}        
        init = tf.initialize_all_variables()
        self.sess.run(init)
        '''
        want to factor in trading fee
        
        so you want average return; assume say that you trade $1000 each time.
        then the threshold needs to be 1.007
        you have the return, e.g. 0.97.
        you then need to multiply this by 1000 and subtract 7.
        or simply subtract 0.007
        
        
        '''

        for i in range(self.steps):
            print(i)
            self.sess.run(self.train_step, feed_dict = feed) 
            self.sess.run(self.normalise_step, feed_dict = feed)      
            print("After %d iterations:" % i)
            total_loss = self.sess.run(self.total_loss, feed_dict=feed)
            print(total_loss)
            print(self.sess.run(self.average_return, feed_dict=feed))


    def test(self, test_actual_buy_prices, test_actual_sell_prices, test_last_sell_prices, test_dependent_variables):
        feed = {self.x: test_dependent_variables, self.actual_buy_prices: test_actual_buy_prices, self.actual_sell_prices: test_actual_sell_prices}                
        print(self.sess.run(self.average_return, feed_dict=feed))
        print(self.sess.run(tf.reduce_sum(self.will_buy), feed_dict=feed))
        print(self.sess.run(self.total_return, feed_dict=feed))
        return self.sess.run(self.predicted_sell_prices, feed_dict=feed), self.sess.run(self.returns, feed_dict=feed)
#         proposed_buy_prices = self.sess.run(self.proposed_buy_prices, feed_dict=feed)
#         actual_buy_prices = self.sess.run(self.actual_buy_prices, feed_dict=feed)
#         print(proposed_buy_prices[np.where(proposed_buy_prices > actual_buy_prices)])
#         print(actual_buy_prices[np.where(proposed_buy_prices > actual_buy_prices)])
#         print(proposed_buy_prices)
#         print(actual_buy_prices)

class NewModel_5:

    def __init__(self, actual_buy_prices, actual_sell_prices, last_sell_prices, dependent_variables, positive = True, include_intercept = False, load_previous=False, weights=None):
        
        '''
        latest idea; 
        you want to detect events where the price will rise
        
        what you are using is previous prices;
        
        or could use a more sophisticated probability distribution
        e.g. you are suggesting two different prices that something can move to
        not measuring accuracy, but cross-entropy;
        or basically just the probability if it is one price + prob if it is the other price, weighted.
        
        will first try using the maximum sell price over a period.
        
        
        
        
        
        '''

        self.steps = 5000
        self.learn_rate = 0.001
        self.dependent_variables = np.array(dependent_variables)
        self.actual_buy_prices_array = np.array(actual_buy_prices)
        self.actual_sell_prices_array = np.array(actual_sell_prices)
        self.last_sell_prices_array = last_sell_prices
        self.positive = positive
        self.include_intercept = include_intercept
        self.one_hot_encodings = []	
        self.other_vectors = []
        self.sess = tf.Session()
        self.x = tf.placeholder(tf.float32, [None, len(self.dependent_variables[0])], name = "x")
        x = rep(0.0, len(self.dependent_variables[0]))
        x[-1] = 1.0
        x = np.array(x, dtype = np.float32)
        x = np.reshape(x, (len(x), 1))
        if load_previous:
          self.W_buy = tf.convert_to_tensor(weights, tf.float32)
        else:
          self.W_buy = tf.Variable(tf.constant(x), name = "W_buy")
        self.predicted_sell_prices = tf.matmul(self.x, self.W_buy)
        self.actual_buy_prices = tf.placeholder(tf.float32, [None, 1])
        self.actual_sell_prices = tf.placeholder(tf.float32, [None, 1])
        self.loss = tf.abs(self.actual_sell_prices - self.predicted_sell_prices)
        self.total_loss = tf.reduce_sum(self.loss)
        fee = 0.0007
        threshold = 1.0 + fee
        self.will_buy = tf.cast(tf.less(self.actual_buy_prices * threshold, self.predicted_sell_prices), tf.float32)
        self.will_not_buy = 1.0 - self.will_buy
        self.hypothetical_returns = self.actual_sell_prices / self.actual_buy_prices
        self.returns = self.will_buy * self.hypothetical_returns
        self.average_return = tf.reduce_sum(self.returns) / tf.reduce_sum(self.will_buy) 
        self.average_return = self.average_return - fee    
        self.total_return = tf.reduce_prod(self.returns + self.will_not_buy)

        
        '''
        now making just accuracy.      
        return is defined by having a predicted sell price; if actual buy price is lower than that, buy it.
        then use actual_sell_price / actual_buy_price as the return.
        '''

    def train(self):     
 
        self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.total_loss)	
        feed = {self.x: self.dependent_variables, self.actual_buy_prices: self.actual_buy_prices_array, self.actual_sell_prices: self.actual_sell_prices_array}        
        init = tf.initialize_all_variables()
        self.sess.run(init)
        '''
        want to factor in trading fee
        
        so you want average return; assume say that you trade $1000 each time.
        then the threshold needs to be 1.007
        you have the return, e.g. 0.97.
        you then need to multiply this by 1000 and subtract 7.
        or simply subtract 0.007
        
        
        '''

        for i in range(self.steps):
            print(i)
            self.sess.run(self.train_step, feed_dict = feed)       
            print("After %d iterations:" % i)
            total_loss = self.sess.run(self.total_loss, feed_dict=feed)
            print(total_loss)
            print(self.sess.run(self.average_return, feed_dict=feed))


    def test(self, test_actual_buy_prices, test_actual_sell_prices, test_last_sell_prices, test_dependent_variables):
        feed = {self.x: test_dependent_variables, self.actual_buy_prices: test_actual_buy_prices, self.actual_sell_prices: test_actual_sell_prices}                
        print(self.sess.run(self.average_return, feed_dict=feed))
        print(self.sess.run(tf.reduce_sum(self.will_buy), feed_dict=feed))
        print(self.sess.run(self.total_return, feed_dict=feed))
        return self.sess.run(self.predicted_sell_prices, feed_dict=feed), self.sess.run(self.returns, feed_dict=feed)
#         proposed_buy_prices = self.sess.run(self.proposed_buy_prices, feed_dict=feed)
#         actual_buy_prices = self.sess.run(self.actual_buy_prices, feed_dict=feed)
#         print(proposed_buy_prices[np.where(proposed_buy_prices > actual_buy_prices)])
#         print(actual_buy_prices[np.where(proposed_buy_prices > actual_buy_prices)])
#         print(proposed_buy_prices)
#         print(actual_buy_prices)


class NewModel_4:

    def __init__(self, actual_buy_prices, actual_sell_prices, last_sell_prices, dependent_variables, positive = True, include_intercept = False, load_previous=False, weights=None):
        self.steps = 30000
        self.learn_rate = 0.001
        self.dependent_variables = np.array(dependent_variables)
        self.actual_buy_prices_array = np.array(actual_buy_prices)
        self.actual_sell_prices_array = np.array(actual_sell_prices)
        self.last_sell_prices_array = last_sell_prices
        self.positive = positive
        self.include_intercept = include_intercept
        self.one_hot_encodings = []	
        self.other_vectors = []
        self.sess = tf.Session()
        self.x = tf.placeholder(tf.float32, [None, len(self.dependent_variables[0])], name = "x")
        x = rep(0.0, len(self.dependent_variables[0]))
        x[-1] = 1.0
        x = np.array(x, dtype = np.float32)
        x = np.reshape(x, (len(x), 1))
        if load_previous:
          self.W_buy = tf.convert_to_tensor(weights, tf.float32)
        else:
          self.W_buy = tf.Variable(tf.constant(x), name = "W_buy")
        self.predicted_sell_prices = tf.matmul(self.x, self.W_buy)
        self.actual_buy_prices = tf.placeholder(tf.float32, [None, 1])
        self.actual_sell_prices = tf.placeholder(tf.float32, [None, 1])
        self.loss = tf.abs(self.actual_sell_prices - self.predicted_sell_prices)
        self.total_loss = tf.reduce_sum(self.loss)
        threshold = 1.000
        self.will_buy = tf.cast(tf.less(self.actual_buy_prices * threshold, self.predicted_sell_prices), tf.float32)
        self.will_not_buy = 1.0 - self.will_buy
        self.hypothetical_returns = self.actual_sell_prices / self.actual_buy_prices
        self.returns = self.will_buy * self.hypothetical_returns
        self.average_return = tf.reduce_sum(self.returns) / tf.reduce_sum(self.will_buy) 
        self.average_return = self.average_return - 0.000    
        self.total_return = tf.reduce_prod(self.returns + self.will_not_buy)

        
        '''
        now making just accuracy.      
        return is defined by having a predicted sell price; if actual buy price is lower than that, buy it.
        then use actual_sell_price / actual_buy_price as the return.
        '''

    def train(self):     
 
        self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.total_loss)	
        feed = {self.x: self.dependent_variables, self.actual_buy_prices: self.actual_buy_prices_array, self.actual_sell_prices: self.actual_sell_prices_array}        
        init = tf.initialize_all_variables()
        self.sess.run(init)
        '''
        want to factor in trading fee
        
        so you want average return; assume say that you trade $1000 each time.
        then the threshold needs to be 1.007
        you have the return, e.g. 0.97.
        you then need to multiply this by 1000 and subtract 7.
        or simply subtract 0.007
        
        
        '''

        for i in range(self.steps):
            print(i)
            self.sess.run(self.train_step, feed_dict = feed)       
            print("After %d iterations:" % i)
            total_loss = self.sess.run(self.total_loss, feed_dict=feed)
            print(total_loss)
            print(self.sess.run(self.average_return, feed_dict=feed))


    def test(self, test_actual_buy_prices, test_actual_sell_prices, test_last_sell_prices, test_dependent_variables):
        feed = {self.x: test_dependent_variables, self.actual_buy_prices: test_actual_buy_prices, self.actual_sell_prices: test_actual_sell_prices}                
        print(self.sess.run(self.average_return, feed_dict=feed))
        print(self.sess.run(tf.reduce_sum(self.will_buy), feed_dict=feed))
        print(self.sess.run(self.total_return, feed_dict=feed))

#         proposed_buy_prices = self.sess.run(self.proposed_buy_prices, feed_dict=feed)
#         actual_buy_prices = self.sess.run(self.actual_buy_prices, feed_dict=feed)
#         print(proposed_buy_prices[np.where(proposed_buy_prices > actual_buy_prices)])
#         print(actual_buy_prices[np.where(proposed_buy_prices > actual_buy_prices)])
#         print(proposed_buy_prices)
#         print(actual_buy_prices)


class NewModel_3:

    def __init__(self, actual_buy_prices, actual_sell_prices, last_sell_prices, dependent_variables, positive = True, include_intercept = False, load_previous=False, weights=None):
        self.steps = 40000
        self.learn_rate = 0.001
        self.dependent_variables = np.array(dependent_variables)
        self.actual_buy_prices_array = np.array(actual_buy_prices)
        self.actual_sell_prices_array = np.array(actual_sell_prices)
        self.last_sell_prices_array = last_sell_prices
        self.positive = positive
        self.include_intercept = include_intercept
        self.one_hot_encodings = []	
        self.other_vectors = []
        self.sess = tf.Session()
        self.x = tf.placeholder(tf.float32, [None, len(self.dependent_variables[0])], name = "x")
        x = rep(0.0, len(self.dependent_variables[0]))
        x[-1] = 1.0
        x = np.array(x, dtype = np.float32)
        x = np.reshape(x, (len(x), 1))
        if load_previous:
          self.W_buy = tf.convert_to_tensor(weights, tf.float32)
        else:
          self.W_buy = tf.Variable(tf.constant(x), name = "W_buy")
        self.predicted_buy_prices = tf.matmul(self.x, self.W_buy)
        self.actual_buy_prices = tf.placeholder(tf.float32, [None, 1])
        self.actual_sell_prices = tf.placeholder(tf.float32, [None, 1])
        margin_method = 'multiply without changing median'
#         margin_method = 'add without adding to median'
        if margin_method == 'multiply':
          self.margin = 0.05
          self.increment = 0.1
          self.maximum_number_of_increments = tf.abs(((self.predicted_buy_prices * (1 + self.margin)) - (self.predicted_buy_prices * (1 - self.margin))) / self.increment) + 1
          self.number1 = tf.minimum(tf.maximum(0.0, ((self.predicted_buy_prices * (1.0 + self.margin)) - self.actual_buy_prices) / self.increment), self.maximum_number_of_increments)
          self.number2 = tf.minimum(tf.maximum(0.0, (self.actual_buy_prices - (self.predicted_buy_prices * (1.0 - self.margin))) / self.increment), self.maximum_number_of_increments)
          self.median_of_the_higher_random_buy_prices = self.actual_buy_prices + tf.maximum(0.0, (self.predicted_buy_prices * (1.0 + self.margin)) - self.actual_buy_prices) / 2
        if margin_method == 'multiply without changing median':
          self.margin = 0.15
          self.increment = 0.1
          self.maximum_number_of_increments = tf.abs(((self.predicted_buy_prices * (1 + self.margin)) - (self.predicted_buy_prices * (1 - self.margin))) / self.increment) + 1
          self.number1 = tf.minimum(tf.maximum(0.0, ((self.predicted_buy_prices * (1.0 + self.margin)) - self.actual_buy_prices) / self.increment), self.maximum_number_of_increments)
          self.number2 = tf.minimum(tf.maximum(0.0, (self.actual_buy_prices - (self.predicted_buy_prices * (1.0 - self.margin))) / self.increment), self.maximum_number_of_increments)
          self.median_of_the_higher_random_buy_prices = self.actual_buy_prices + tf.maximum(0.0, (self.predicted_buy_prices) - self.actual_buy_prices) / 2
        if margin_method == 'add':
          self.margin = 3.0
          self.increment = 0.1
          self.maximum_number_of_increments = tf.abs((self.predicted_buy_prices + self.margin) - (self.predicted_buy_prices - self.margin)) / self.increment 
          self.number1 = tf.minimum(tf.maximum(0.0, (self.predicted_buy_prices + self.margin - self.actual_buy_prices) / self.increment), self.maximum_number_of_increments)
          self.number2 = tf.minimum(tf.maximum(0.0, (self.actual_buy_prices - (self.predicted_buy_prices - self.margin)) / self.increment), self.maximum_number_of_increments)
          self.median_of_the_higher_random_buy_prices = self.actual_buy_prices + tf.maximum(0.0, (self.predicted_buy_prices + self.margin - self.actual_buy_prices) / 2)
        if margin_method == 'add without changing median':
          self.margin = 3.0
          self.increment = 0.1
          self.maximum_number_of_increments = tf.abs((self.predicted_buy_prices + self.margin) - (self.predicted_buy_prices - self.margin)) / self.increment 
          self.number1 = tf.minimum(tf.maximum(0.0, (self.predicted_buy_prices + self.margin - self.actual_buy_prices) / self.increment), self.maximum_number_of_increments)
          self.number2 = tf.minimum(tf.maximum(0.0, (self.actual_buy_prices - (self.predicted_buy_prices - self.margin)) / self.increment), self.maximum_number_of_increments)
          self.median_of_the_higher_random_buy_prices = self.actual_buy_prices + tf.maximum(0.0, (self.predicted_buy_prices - self.actual_buy_prices) / 2)


        self.sell_greater_than_buy = tf.cast(tf.greater(self.actual_sell_prices, self.actual_buy_prices), tf.float32)
        self.sell_less_than_buy = 1 - self.sell_greater_than_buy
        self.if_sell_is_greater_than_buy = (self.number1 / (self.number1 + self.number2)) * (self.actual_buy_prices / self.median_of_the_higher_random_buy_prices)
        self.if_sell_is_greater_than_buy = self.if_sell_is_greater_than_buy + (self.number2 / (self.number1 + self.number2) * (self.actual_buy_prices / self.actual_sell_prices))        
        self.if_sell_is_less_than_buy = (self.number1 / (self.number1 + self.number2)) * (self.actual_sell_prices / self.median_of_the_higher_random_buy_prices)
        self.if_sell_is_less_than_buy = self.if_sell_is_less_than_buy + (self.number2 / (self.number1 + self.number2))
        self.loss = (self.sell_greater_than_buy * self.if_sell_is_greater_than_buy) + (self.sell_less_than_buy * self.if_sell_is_less_than_buy)
        self.b = tf.reduce_sum(self.if_sell_is_greater_than_buy)
        self.loss = tf.log(self.loss) * -1.0 
        m = tf.log(self.sell_greater_than_buy + self.sell_less_than_buy)      
#         self.total_loss = tf.reduce_sum(self.loss)    
        self.total_loss = tf.reduce_mean(self.loss)    


    def train(self):     
 
        self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.total_loss)	
        feed = {self.x: self.dependent_variables, self.actual_buy_prices: self.actual_buy_prices_array, self.actual_sell_prices: self.actual_sell_prices_array}        
        init = tf.initialize_all_variables()
        self.sess.run(init)
        
        self.proposed_buy_prices = self.predicted_buy_prices 
        self.return_1 = tf.cast(tf.greater_equal(self.proposed_buy_prices, self.actual_buy_prices), tf.float32)
        self.return_2 = self.return_1 * (self.actual_sell_prices / self.proposed_buy_prices)
        self.return_3 = 1.0 - self.return_1
        self.returns = self.return_2 + self.return_3
#         self.returns = tf.reduce_mean(self.returns)
        self.returns = tf.reduce_mean(self.returns)
        
        for i in range(self.steps):
            print(i)
            self.sess.run(self.train_step, feed_dict = feed)       
            print("After %d iterations:" % i)
            total_loss = self.sess.run(self.total_loss, feed_dict=feed)
            print(total_loss)
            print(self.sess.run(self.returns, feed_dict=feed))
#             print(self.sess.run(self.predicted_buy_prices, feed_dict=feed))
#             loss = self.sess.run(self.b, feed_dict=feed))
            if i%1000 == 0:
              print(self.sess.run(self.predicted_buy_prices, feed_dict=feed))
            if np.isnan(total_loss):
              exit(0)

    def test(self, test_actual_buy_prices, test_actual_sell_prices, test_last_sell_prices, test_dependent_variables):
        feed = {self.x: test_dependent_variables, self.actual_buy_prices: test_actual_buy_prices, self.actual_sell_prices: test_actual_sell_prices}                
        self.proposed_buy_prices = self.predicted_buy_prices 
        self.return_1 = tf.cast(tf.greater_equal(self.proposed_buy_prices, self.actual_buy_prices), tf.float32)
        self.return_2 = self.return_1 * (self.actual_sell_prices / self.proposed_buy_prices)
        self.return_3 = 1.0 - self.return_1
        self.returns = self.return_2 + self.return_3
        self.returns = tf.reduce_mean(self.returns)
#         self.returns = tf.reduce_prod(self.returns)
        print(self.sess.run(self.returns, feed_dict=feed))
        proposed_buy_prices = self.sess.run(self.proposed_buy_prices, feed_dict=feed)
        actual_buy_prices = self.sess.run(self.actual_buy_prices, feed_dict=feed)
        print(proposed_buy_prices[np.where(proposed_buy_prices > actual_buy_prices)])
        print(actual_buy_prices[np.where(proposed_buy_prices > actual_buy_prices)])
        print(proposed_buy_prices)
        print(actual_buy_prices)

#         print(np.concatenate([np.delete(actual_buy_prices, [0,1,2]), [1.0,2.0,3.0]])[np.where(proposed_buy_prices > actual_buy_prices)])
#         print(np.concatenate([[1.0], actual_buy_prices])[np.where(proposed_buy_prices > actual_buy_prices)])


    
class NewModel_2:

    def __init__(self, actual_buy_prices, actual_sell_prices, last_sell_prices, dependent_variables, positive = True, include_intercept = False):
        self.steps = 120000
        self.learn_rate = 0.001
        self.dependent_variables = np.array(dependent_variables)
        self.actual_buy_prices_array = np.array(actual_buy_prices)
        self.actual_sell_prices_array = np.array(actual_sell_prices)
        self.last_sell_prices_array = last_sell_prices
        self.positive = positive
        self.include_intercept = include_intercept
        self.one_hot_encodings = []	
        self.other_vectors = []
        self.sess = tf.Session()
    
    def create_linear_layers(self, load_previous=False):
        self.x = tf.placeholder(tf.float32, [None, len(self.dependent_variables[0])], name = "x")
        x = rep(0.0, len(self.dependent_variables[0]))
        x[-1] = 1.0
        x = np.array(x, dtype = np.float32)
        x = np.reshape(x, (len(x), 1))
        self.W_buy = tf.Variable(tf.constant(x), name = "W_buy")
        self.predicted_buy_prices = tf.matmul(self.x, self.W_buy)
    
    def train(self, load_previous=False):     
        self.actual_buy_prices = tf.placeholder(tf.float32, [None, 1])
        self.actual_sell_prices = tf.placeholder(tf.float32, [None, 1])
        self.create_linear_layers(load_previous)
        margin_method = 'multiply'
#         margin_method = 'add without adding to median'
        if margin_method == 'multiply':
          self.margin = 0.05
          self.increment = 0.1
          self.maximum_number_of_increments = tf.abs(((self.predicted_buy_prices * (1 + self.margin)) - (self.predicted_buy_prices * (1 - self.margin))) / self.increment) + 1
          self.number1 = tf.minimum(tf.maximum(0.0, ((self.predicted_buy_prices * (1.0 + self.margin)) - self.actual_buy_prices) / self.increment), self.maximum_number_of_increments)
          self.number2 = tf.minimum(tf.maximum(0.0, (self.actual_buy_prices - (self.predicted_buy_prices * (1.0 - self.margin))) / self.increment), self.maximum_number_of_increments)
          self.median_of_the_higher_random_buy_prices = self.actual_buy_prices + tf.maximum(0.0, (self.predicted_buy_prices * (1.0 + self.margin)) - self.actual_buy_prices) / 2
        if margin_method == 'add':
          self.margin = 3.0
          self.increment = 0.1
          self.maximum_number_of_increments = tf.abs((self.predicted_buy_prices + self.margin) - (self.predicted_buy_prices - self.margin)) / self.increment 
          self.number1 = tf.minimum(tf.maximum(0.0, (self.predicted_buy_prices + self.margin - self.actual_buy_prices) / self.increment), self.maximum_number_of_increments)
          self.number2 = tf.minimum(tf.maximum(0.0, (self.actual_buy_prices - (self.predicted_buy_prices - self.margin)) / self.increment), self.maximum_number_of_increments)
          self.median_of_the_higher_random_buy_prices = self.actual_buy_prices + tf.maximum(0.0, (self.predicted_buy_prices + self.margin - self.actual_buy_prices) / 2)
        if margin_method == 'add without adding to median':
          self.margin = 3.0
          self.increment = 0.1
          self.maximum_number_of_increments = tf.abs((self.predicted_buy_prices + self.margin) - (self.predicted_buy_prices - self.margin)) / self.increment 
          self.number1 = tf.minimum(tf.maximum(0.0, (self.predicted_buy_prices + self.margin - self.actual_buy_prices) / self.increment), self.maximum_number_of_increments)
          self.number2 = tf.minimum(tf.maximum(0.0, (self.actual_buy_prices - (self.predicted_buy_prices - self.margin)) / self.increment), self.maximum_number_of_increments)
          self.median_of_the_higher_random_buy_prices = self.actual_buy_prices + tf.maximum(0.0, (self.predicted_buy_prices - self.actual_buy_prices) / 2)


        self.sell_greater_than_buy = tf.cast(tf.greater(self.actual_sell_prices, self.actual_buy_prices), tf.float32)
        self.sell_less_than_buy = 1 - self.sell_greater_than_buy
        self.if_sell_is_greater_than_buy = (self.number1 / (self.number1 + self.number2)) * (self.actual_buy_prices / self.median_of_the_higher_random_buy_prices)
        self.if_sell_is_greater_than_buy = self.if_sell_is_greater_than_buy + (self.number2 / (self.number1 + self.number2) * (self.actual_buy_prices / self.actual_sell_prices))        
        self.if_sell_is_less_than_buy = (self.number1 / (self.number1 + self.number2)) * (self.actual_sell_prices / self.median_of_the_higher_random_buy_prices)
        self.if_sell_is_less_than_buy = self.if_sell_is_less_than_buy + (self.number2 / (self.number1 + self.number2))
        self.loss = (self.sell_greater_than_buy * self.if_sell_is_greater_than_buy) + (self.sell_less_than_buy * self.if_sell_is_less_than_buy)
        self.b = tf.reduce_sum(self.if_sell_is_greater_than_buy)
        self.loss = tf.log(self.loss) * -1.0 
        m = tf.log(self.sell_greater_than_buy + self.sell_less_than_buy)
        
        
        self.total_loss = tf.reduce_sum(self.loss) 
        self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.total_loss)	
        feed = {self.x: self.dependent_variables, self.actual_buy_prices: self.actual_buy_prices_array, self.actual_sell_prices: self.actual_sell_prices_array}        
        init = tf.initialize_all_variables()
        self.sess.run(init)
        
        self.proposed_buy_prices = self.predicted_buy_prices 
        self.return_1 = tf.cast(tf.greater_equal(self.proposed_buy_prices, self.actual_buy_prices), tf.float32)
        self.return_2 = self.return_1 * (self.actual_sell_prices / self.proposed_buy_prices)
        self.return_3 = 1.0 - self.return_1
        self.returns = self.return_2 + self.return_3
#         self.returns = tf.reduce_mean(self.returns)
        self.returns = tf.reduce_prod(self.returns)
        
        for i in range(self.steps):
            print(i)
            self.sess.run(self.train_step, feed_dict = feed)       
            print("After %d iterations:" % i)
            total_loss = self.sess.run(self.total_loss, feed_dict=feed)
            print(total_loss)
            print(self.sess.run(self.returns, feed_dict=feed))
#             print(self.sess.run(self.predicted_buy_prices, feed_dict=feed))
#             loss = self.sess.run(self.b, feed_dict=feed))
            if i%1000 == 0:
              print(self.sess.run(self.predicted_buy_prices, feed_dict=feed))
            if np.isnan(total_loss):
              exit(0)



class NewModel1:

    def __init__(self, actual_buy_prices, actual_sell_prices, last_sell_prices, dependent_variables, positive = True, include_intercept = False):
        self.steps = 120000
        self.learn_rate = 0.001
        self.dependent_variables = np.array(dependent_variables)
        self.actual_buy_prices_array = np.array(actual_buy_prices)
        self.actual_sell_prices_array = np.array(actual_sell_prices)
        self.last_sell_prices_array = last_sell_prices
        self.positive = positive
        self.include_intercept = include_intercept
        self.one_hot_encodings = []	
        self.other_vectors = []
        self.sess = tf.Session()
    
    def create_linear_layers(self, load_previous=False):
        self.x = tf.placeholder(tf.float32, [None, len(self.dependent_variables[0])], name = "x")
        x = rep(0.0, len(self.dependent_variables[0]))
        x[-1] = 1.0
        x = np.array(x, dtype = np.float32)
        x = np.reshape(x, (len(x), 1))
        self.W_buy = tf.Variable(tf.constant(x), name = "W_buy")
        self.predicted_buy_prices = tf.matmul(self.x, self.W_buy)
    
    def train(self, load_previous=False):     
        self.actual_buy_prices = tf.placeholder(tf.float32, [None, 1])
        self.actual_sell_prices = tf.placeholder(tf.float32, [None, 1])
        self.create_linear_layers(load_previous)
        self.margin = 3.0
        self.increment = 0.1
        self.sell_greater_than_buy = tf.cast(tf.greater(self.actual_sell_prices, self.actual_buy_prices), tf.float32)
        self.sell_less_than_buy = 1 - self.sell_greater_than_buy
        self.number1 = tf.minimum(tf.maximum(0.0, (self.predicted_buy_prices + self.margin - self.actual_buy_prices) / self.increment), self.margin * 2 / self.increment)
        self.number2 = tf.minimum(tf.maximum(0.0, (self.actual_buy_prices - self.predicted_buy_prices + self.margin) / self.increment), self.margin * 2 / self.increment)
        self.median_of_the_higher_random_buy_prices = self.actual_buy_prices + tf.maximum(0.0, self.predicted_buy_prices - self.actual_buy_prices) / 2
        
        self.if_sell_is_greater_than_buy = (self.number1 / (self.number1 + self.number2)) * (self.actual_buy_prices / self.median_of_the_higher_random_buy_prices)
        self.if_sell_is_greater_than_buy = self.if_sell_is_greater_than_buy + (self.number2 / (self.number1 + self.number2) * (self.actual_buy_prices / self.actual_sell_prices))
        
        self.if_sell_is_less_than_buy = (self.number1 / (self.number1 + self.number2)) * (self.actual_sell_prices / self.median_of_the_higher_random_buy_prices)
        self.if_sell_is_less_than_buy = self.if_sell_is_less_than_buy + (self.number2 / (self.number1 + self.number2))
        self.b = (self.actual_sell_prices / self.median_of_the_higher_random_buy_prices)
        self.loss = (self.sell_greater_than_buy * self.if_sell_is_greater_than_buy) + (self.sell_less_than_buy * self.if_sell_is_less_than_buy)
        self.loss_before = (self.sell_greater_than_buy * self.if_sell_is_greater_than_buy)
        self.loss_before2 = (self.sell_less_than_buy * self.if_sell_is_less_than_buy)
        self.loss = tf.log(self.loss) * -1.0 
        m = tf.log(self.sell_greater_than_buy + self.sell_less_than_buy)
        
        
        self.total_loss = tf.reduce_sum(self.loss) 
        self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.total_loss)	
        feed = {self.x: self.dependent_variables, self.actual_buy_prices: self.actual_buy_prices_array, self.actual_sell_prices: self.actual_sell_prices_array}        
        init = tf.initialize_all_variables()
        self.sess.run(init)
        
#         self.proposed_buy_prices = tf.maximum(self.increment, self.predicted_buy_prices + tf.random.uniform(np.shape(self.actual_buy_prices_array), minval = -1.0 * self.margin/10.0, maxval= self.margin/10.0, dtype=tf.dtypes.float32, seed=None, name=None))
        self.proposed_buy_prices = self.predicted_buy_prices 
        self.return_1 = tf.cast(tf.greater_equal(self.proposed_buy_prices, self.actual_buy_prices), tf.float32)
        self.return_2 = self.return_1 * (self.actual_sell_prices / self.proposed_buy_prices)
        self.return_3 = 1.0 - self.return_1
        self.returns = self.return_2 + self.return_3
#         self.returns = tf.reduce_mean(self.returns)
        self.returns = tf.reduce_prod(self.returns)
        
        for i in range(self.steps):
            print(i)
            self.sess.run(self.train_step, feed_dict = feed)       
            print("After %d iterations:" % i)
            print(self.sess.run(self.total_loss, feed_dict=feed))
            print(self.sess.run(self.returns, feed_dict=feed))
            if i%1000 == 0:
              print(self.sess.run(self.W_buy))
            '''
            how do you now calculate the return?
            
            the model is outputting its predicted buy prices;
            you need some separate tensors for calculating this
        
            a tensor comparing the predicted buy price with the actual buy price;
            1 if it is greater or equal, zero otherwise.
            then multiply this by actual_sell_price / predicted_buy_price.
            
            actually it needs to take into account the margin as well, include that 
            
            now want to rewrite it so that margin is a constant multiplied by the predicted_buy_price, rather than added
            
            there still seems to be a disconnect between the loss function and returns.
            
            problems here;
            what exactly is the loss function calculating?  
            why is this still hard to translate into a strategy?
            
            
            
            '''

#             y = self.sess.run(self.loss, feed_dict=feed)
#             a = self.sess.run(self.b, feed_dict=feed)
#             for j in range(5,7):
#               print(y[j], a[j])
              
#             print(self.sess.run(self.predicted_buy_prices, feed_dict=feed))
#             print(self.sess.run(self.actual_buy_prices, feed_dict=feed))
#             print(self.sess.run(self.actual_sell_prices, feed_dict=feed))
# 
#             print(self.sess.run(self.loss_before, feed_dict=feed))



'''

latest version;
you have predicted returns across a range of proposed buy and sell prices.
you are trying to maximise the accuracy of those predictions. the loss is abs(log(actual_return / predicted_return)).
but you have an approximate way of calculating this across a range of prices.
e.g.
say that you are predicting a buy price of 10 and sell price of 20.
the actual buy price is 12 not 10.
then you will get 1 where you thought you would get 2; and 1 where you thought you would get 2/1.1
so the loss for those would be 
1/2 and 1/(2/1.1)

sum of logs of 1 /  (predicted_return / predicted_buy_price)
for every value of predicted_buy_price that is lower than actual_buy_price.
so you have predicted_buy_price and actual_buy_price.
one component of the loss function so far is:
sum of logs of 1 /  (predicted_return / predicted_buy_price)
for every incremental value of predicted_buy_price that is lower than actual_buy_price.
not sure how to make that a tensorflow operation.
find an approximate function for this.




for other values;
you are predicting a buy_price.  if it is above the actual_buy_price, then actually it is accurate, if the predicted sell price is also accurate.

what about the predicted sell price? do that next.


'''








'''
latest version:

you have predicted_buy_prices and predicted_sell_prices,
and actual_buy_prices and actual_sell_prices

the loss is:
abs(log ideal_return/actual_return), for cases when ideal_return > 1 or actual_return > 1.
when they are both under 1, then there is no loss.
and you are trying to minimise the loss.

ideal_return = actual_sell_prices/actual_buy_prices
actual_return = min(actual_sell_prices, predicted_sell_prices) / predicted_buy_prices
when predicted_buy_prices > actual_buy_prices; 1 otherwise.













the task is to place a buy bid that may be accepted and give you good returns.
there are different ways of doing this;
just a certain proportion of the current price apparently works to some extent.

you may also have some reason to think the price will go up a lot, e.g. if the fundamentals suggest they may.
in which case you can have a separate suggestion of a bidding price from that module.



new version;
you have the data.
you have a suggested buy price and suggested sell price (some time later)
the model outputs the predicted return of doing that; which could be 1, if it thinks it will be unable to 
buy it at that price.
training the model is therefore training it in accurately forecasting the return.

so for this i need;

the actual buy prices and sell prices; this i have.
the actual return of suggested buy and sell prices; so i need to work out how i am doing it.
this is calculated as min(suggested sell price, actual sell price) / suggested buy price
if suggested buy price > actual buy price; otherwise it is 1.
the model needs to predict the buy price.  if the suggested buy price is lower than its prediction,
then it has to provide a predicted return of 1.
the model also needs to predict the sell price.  if the suggested sell price is lower than the predicted sell price,
then the predicted return is the suggested sell price/suggested buy price.
if the suggested sell price is higher than the predicted sell price, then the predicted return is 
predicted sell price/suggested buy price.

sometimes it will wrongly predict 1 for the return if it wrongly thinks the suggested buy price is too low.
how does it then update?
it has the predicted price and the suggested price; 
the function is something like
max(predicted price, suggested price)
let's say that the actual return is greater than one.


the main thing that the model just needs to do is predict the buy and sell prices.
if it does that accurately, then it can predict the return.
the question is how 'accurately' is defined; the loss function.
the model is predicting the return.  it should be penalised for missing opportunities, or
placing bids which will make it lose money.
so it is the overall returns which matter.
the model calculates the returns for each suggested buy and sell price on each day.
it then finds the maximum, and that is the one that is considering for that day.
it multiplies them together to give the total return.
you then have the actual returns which a perfectly accurate model could get;
which is found by using the actual prices.
the difference between them is the loss.


so what is the structure of the model?

you have 
self.actual_buy_prices_array
self.actual_sell_prices_array
self.dependent_variables

self.predicted_buy_prices
self.predicted_sell_prices

then you have a range of suggested buy prices
and a range of suggested sell prices
you 

'''




class LinearRegression:
    steps = 2000
    learn_rate = 0.001
    def __init__(self, actual_buy_prices, actual_sell_prices, last_sell_prices, dependent_variables, positive = True, include_intercept = False):
        self.dependent_variables = np.array(dependent_variables)
        self.actual_buy_prices_array = np.array(actual_buy_prices)
        self.actual_sell_prices_array = np.array(actual_sell_prices)
        self.last_sell_prices_array = last_sell_prices
        self.positive = positive
        self.include_intercept = include_intercept
        self.one_hot_encodings = []	
        self.other_vectors = []
        self.sess = tf.Session()
    
    def create_linear_layers(self, load_previous=False):
        self.x = tf.placeholder(tf.float32, [None, len(self.dependent_variables[0])], name = "x")
        if not load_previous:
#             self.W_buy = tf.Variable(tf.zeros([len(self.dependent_variables[0]), 1]), name = "W_buy")
#             self.W_sell = tf.Variable(tf.zeros([len(self.dependent_variables[0]), 1]), name = "W_sell")
            
            x = rep(0.01, len(self.dependent_variables[0]))
            x[-1] = 0.01
            x = np.array(x, dtype = np.float32)
            x = np.reshape(x, (len(x), 1))
            self.W_buy = tf.Variable(tf.constant(x), name = "W_buy")
            self.W_sell = tf.Variable(tf.constant(x), name = "W_sell")

        self.predicted_buy_prices = tf.matmul(self.x, self.W_buy)
        self.predicted_sell_prices = tf.matmul(self.x, self.W_sell)
    
    def train(self, load_previous=False):     
        self.actual_buy_prices = tf.placeholder(tf.float32, [None, 1])
        self.actual_sell_prices = tf.placeholder(tf.float32, [None, 1])
        self.create_linear_layers(load_previous)
        self.ideal_return = self.actual_sell_prices / self.actual_buy_prices
        self.actual_return = tf.minimum(self.actual_sell_prices, self.predicted_sell_prices) / tf.minimum(self.actual_buy_prices, self.predicted_buy_prices)
        self.actual_return_filter = tf.cast(tf.greater_equal(self.predicted_buy_prices, self.actual_buy_prices), dtype=tf.float32)
        self.actual_return = self.actual_return - 1
        self.actual_return = tf.multiply(self.actual_return, self.actual_return_filter)
        self.actual_return = self.actual_return + 1
        self.total_ideal_return = tf.reduce_prod(self.ideal_return)
        self.total_actual_return = tf.reduce_prod(self.actual_return)   
        
        '''
        there is an error in the way that actual_return is being calculated
        if the predicted_buy_price is lower than the actual_buy_price, somehow you need to give a gradient that encourages
        the model to move up the predicted_buy_price in cases when there would be a positive return.
        at the moment actual_return is set to 1 in that case, and ideal_return is greater than 1.  
        but it is not clear that there is a gradient 
        since actual_return_filter is 0.  
        so it should not be 0.
    
        you could have something called 'putative_return'.  so if the predicted_buy_price is
        lower than the actual_buy_price, you set the 
        
        
        the loss can be something like how far away you are from having had a buy_price that would be accepted that would get a good return.
        there can be an extra loss term, which is just how far actual_price is from predicted_buy_price for cases where the return is positive.
        and presumably greater for when the return is bigger.  so something like 
        ((predicted_buy_price - actual_buy_price) * ideal_return)  * loss_filter.
        so this is an extra loss term.
        '''
        
        self.loss = tf.log(self.ideal_return / self.actual_return) 
    
        self.loss_filter = tf.maximum(self.ideal_return, self.actual_return)
        self.loss_filter = tf.cast(tf.greater(self.loss_filter, 1), dtype=tf.float32)
        
        self.loss = tf.multiply(self.loss, self.loss_filter)

        self.loss = tf.reduce_sum(self.loss)

        self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss)	
        feed = {self.x: self.dependent_variables, self.actual_buy_prices: self.actual_buy_prices_array, self.actual_sell_prices: self.actual_sell_prices_array}        
        init = tf.initialize_all_variables()
        self.sess.run(init)
        for i in range(self.steps):
            print(i)
            self.sess.run(self.train_step, feed_dict = feed)       
            print("After %d iterations:" % i)
#             print("W_buy: %s" % self.sess.run(self.W_buy))
#             print("W_sell: %s" % self.sess.run(self.W_sell))
            print(self.sess.run(self.total_ideal_return, feed_dict = feed))
            print(self.sess.run(self.total_actual_return, feed_dict = feed))
            print(self.sess.run(self.actual_buy_prices, feed_dict = feed))
            print(self.sess.run(self.predicted_buy_prices, feed_dict = feed))
#             print(self.sess.run(self.actual_sell_prices, feed_dict = feed))
#             print(self.sess.run(self.predicted_sell_prices, feed_dict = feed))
#             print(self.sess.run(self.loss, feed_dict = feed))
#             print(self.dependent_variables)
 

            

#     def predict(self, load_previous = True, already_initialised = False):
#         if not already_initialised:
#             self.create_linear_layers(load_previous)
#             init = tf.initialize_all_variables()
#             self.sess.run(init)
#         self.feed = {}
#         self.feed[self.x] = self.dependent_variables
#         print(str(self.sess.run(self.W)))
#         print("prediction: " + str(self.sess.run(self.y, feed_dict = self.feed)[-1]) )    
#         print(str(self.sess.run(self.W)))
#         return self.sess.run(self.y, feed_dict = self.feed)
#         
#     def save_weights(self, filename):
#         file = open(filename, 'w')
#         weights = self.sess.run(self.W)
#         weights = [str(x) for x in weights]
#         file.write('\t'.join(weights))
#     
#     def load_weights(self, filename):
#         file = open(filename, 'r').read()
#         weights = file.split('\t')
#         self.W = tf.Variable(tf.constant(np.array([eval(x) for x in weights], dtype=np.float32)), name = "W")
#         print(self.W)
#         print(weights)
# 
#     def produce_prediction_matrix(self, number_of_days, days_in_advance, load_previous = True, already_initialised = False):
#         prediction_matrix = self.predict(load_previous=load_previous, already_initialised=already_initialised)
#         prediction_matrix = np.ndarray.flatten(prediction_matrix)
#         print(prediction_matrix)
#         prediction_matrix = np.concatenate((rep(np.nan, number_of_days+days_in_advance), prediction_matrix))
#         return prediction_matrix
#         
# '''
# prediction_matrix needs to begin with np.nan for every day in days_in_advance + days being used
# then you have the output of the prediction, so probably just using predict()
# 
# 
# 
# '''  
        
class SequenceLinearRegression(LinearRegression):
	def __init__(self, sequences, window_size, offset = 0, positive = True, include_intercept = False, use_padding = True):
		self.window_size = window_size
		self.offset = offset
		self.sequences = sequences
		self.positive = positive
		self.include_intercept = include_intercept
		dependent_variables = []
		target_variable = []
		for sequence in self.sequences:
			normalised_sequence = []
			for n in range(1, len(sequence)):
				normalised_sequence.append([sequence[n] / sequence[n-1]])
				to_append = []
				for number in range(n - window_size, n - offset):
					if number < 0: 
						if use_padding == True:
							to_append.append(sequence[0])
					else:
						to_append.append(sequence[number])
				normaliser = to_append[-1]
				to_append = np.array(to_append)				
				to_append = to_append / normaliser
				dependent_variables.append(to_append)
			target_variable = target_variable + normalised_sequence
		self.target = np.array(target_variable)
		self.dependent_variables = np.array(dependent_variables)
		self.one_hot_encodings = []

class SequenceLinearRegressionIncludingTimestamps(SequenceLinearRegression):
    def include_timestamps(self, timestamps):    
        self.timestamps = timestamps
        self.calendar_months = [find_calendar_month(x) for x in timestamps]
        self.days_of_the_week = [find_day_of_the_week(x) for x in timestamps]
        calendar_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
        				'Oct', 'Nov', 'Dec']
        days_of_the_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
        self.calendar_months_encoding = OneHotEncoder(self.calendar_months, calendar_months).encode()
        self.days_of_the_week_encoding = OneHotEncoder(self.days_of_the_week, days_of_the_week).encode()
        self.include_one_hot_encoding(self.calendar_months_encoding)
        self.include_one_hot_encoding(self.days_of_the_week_encoding)

class SequenceLinearRegressionIncludingConvolutionalNetwork(SequenceLinearRegression):
    def include_timestamps(self, timestamps):    
        self.timestamps = timestamps
        self.calendar_months = [find_calendar_month(x) for x in timestamps]
        self.days_of_the_week = [find_day_of_the_week(x) for x in timestamps]
        calendar_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
        				'Oct', 'Nov', 'Dec']
        days_of_the_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
        self.calendar_months_encoding = OneHotEncoder(self.calendar_months, calendar_months).encode()
        self.days_of_the_week_encoding = OneHotEncoder(self.days_of_the_week, days_of_the_week).encode()
        self.include_one_hot_encoding(self.calendar_months_encoding)
        self.include_one_hot_encoding(self.days_of_the_week_encoding)

    def create_convolutional_network(self):
        self.cnn_input_layer = tf.placeholder(tf.float32, [len(self.dependent_variables), len(self.dependent_variables[0])])
        self.cnn_input_layer_reshaped = tf.reshape(self.cnn_input_layer, [-1, len(self.dependent_variables[0]), 1])
        self.cnn_conv1_filters = 5
        self.cnn_conv1 = tf.layers.conv1d(self.cnn_input_layer_reshaped, filters = self.cnn_conv1_filters, kernel_size = 2, padding = "same", activation = tf.nn.relu)
        self.cnn_last = self.cnn_conv1
        number_of_cnn_outputs = 1
        cnn_last_shape = tf.shape(self.cnn_last)
        cnn_output_length = len(self.dependent_variables[0]) * self.cnn_conv1_filters   
        self.cnn_last_flat = tf.reshape(self.cnn_last, [-1, cnn_output_length])
        self.cnn_output = tf.layers.dense(self.cnn_last_flat, units = number_of_cnn_outputs, activation = tf.nn.sigmoid)
        self.cnn_coefficients =  tf.Variable(tf.zeros([number_of_cnn_outputs, 1]))
        self.cnn_to_add = tf.matmul(self.cnn_output, self.cnn_coefficients)
        self.cnn_to_add = self.cnn_to_add - tf.reduce_mean(self.cnn_to_add)
        self.y = self.y + self.cnn_to_add 
	
    def train(self):      
        datapoint_size = len(self.target)
        batch_size = datapoint_size
        self.create_linear_layers()
        self.create_one_hot_layers()
        self.create_convolutional_network()        
        self.y_ = tf.placeholder(tf.float32, [None, 1])
        self.cost = tf.reduce_mean(tf.square(self.y_ - self.y))
        self.cost_sum = tf.summary.scalar("cost", self.cost)
        self.train_step = tf.train.GradientDescentOptimizer(self.learn_rate).minimize(self.cost)	
        self.clip_op = tf.assign(self.W, tf.clip_by_value(self.W, 0, np.infty))
        self.reduction_ops = {}
        for i in range(len(self.one_hot_encodings)):
            mean = tf.reduce_mean(self.one_hot_encodings_variables[str(i)])
            self.reduction_ops[str(i)] = tf.assign(self.one_hot_encodings_variables[str(i)], self.one_hot_encodings_variables[str(i)] - mean)
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        for i in range(self.steps):
            print(i)
            if datapoint_size == batch_size:
                batch_start_idx = 0
            elif datapoint_size < batch_size:
                raise ValueError("datapoint_size: %d, must be greater than batch_size: %d" % (datapoint_size, batch_size))
            else:
                batch_start_idx = (i * batch_size) % (datapoint_size - batch_size)
            batch_end_idx = batch_start_idx + batch_size
            batch_x = self.dependent_variables[batch_start_idx:batch_end_idx]
            batch_y = self.target[batch_start_idx:batch_end_idx]
            feed = {self.x: batch_x, self.y_: batch_y, self.cnn_input_layer: batch_x}
            for j in range(len(self.one_hot_encodings)):
                to_feed = self.one_hot_encodings[j].encoding[batch_start_idx:batch_end_idx]
                feed[self.one_hot_encodings_placeholders[str(j)]] = to_feed
            sess.run(self.train_step, feed_dict = feed)
            if self.positive:
                sess.run(self.clip_op)
            for j in range(len(self.one_hot_encodings)):
            	sess.run(self.reduction_ops[str(j)])
            print("After %d iterations:" % i)
            print("W: %s" % sess.run(self.W))
            if self.include_intercept:
                print("b: %f" % sess.run(self.b))
            for j in range(len(self.one_hot_encodings)):
                print("one_hot_encodings_variable" + str(j) + " : %s" % sess.run(self.one_hot_encodings_variables[str(j)]))
            print("Convolutional coefficient: %s" % sess.run(self.cnn_coefficients))



        
        
'''
(notes)
buy_price_filter = max(min(predicted_buy_price - actual_buy_price, 0) * 100, -1) *  -1 

sell_price_filter = max(min(actual_sell_price - predicted_sell_price, 0) * 100, -1) * -1

final_sell_price = (last_sell_price * (1-sell_price_filter)) + (sell_price_filter*predicted_sell_price)

predicted_return = final_sell_price / predicted_buy_price

returns = (buy_price_filter * predicted_return) + (1 - buy_price_filter)

penalty = max(-100, min(0, predicted_buy_price - actual_buy_price) * -1) 

the returns are multiplied together.  the penalties are added together.  you are trying to maximise both.  so you need to check the sign; adam optimizer. maximise.

        self.buy_price_filter = tf.max
        self.sell_price_filter = tf.maximum(tf.minimum)

tf.maximum takes two arrays and returns the maximum element wise
so you want tf.maximum()

you are also finding an array of buy prices actually.
        self.buy_price_filter = tf.maximum(tf.minimum(tf.subtract(predicted_buy_prices, actual_buy_prices), 0) * 100, -1) * -1
        self.sell_price_filter = tf.maximum(tf.minimum(tf.subtract(actual_sell_prices, predicted_sell_prices), 0) * 100, -1) * -1
        self.final_sell_prices = tf.add(tf.multiply(last_sell_prices, tf.subtract(1, sell_price_filter)), tf.multiply(sell_price_filter, predicted_sell_prices)
        self.predicted_returns = tf.divide(final_sell_prices, predicted_buy_prices)
        self.returns = tf.add(tf.divide(buy_price_filter, self.predicted_returns), tf.subtract(1, buy_price_filter))
        self.penalty = tf.multiply(tf.maximum(np.array([-100]), tf.minimum(np.array([0]), tf.subtract(predicted_buy_prices, actual_buy_prices))), -1)
        self.total_returns = tf.reduce_prod(self.returns)
        self.total_penalty = tf.reduce_sum(penalty)
        
        

'''






