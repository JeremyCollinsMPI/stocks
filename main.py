from Model import *
from general import *
from get_data import *
import pandas as pd
from get_companies import *
import sys
from find_returns import *
from fundamental_model import *
from datetime import date, datetime
import os

def make_arrays(data, companies, number_of_days, buy_window = 1, sell_window = 1, days_in_advance = 0):
  actual_buy_prices = []
  actual_sell_prices = []
  last_sell_prices = []
  dependent_variables = []
  for company in companies:
    column = data[company]
    column = pd.Series.dropna(column)
    if len(column) < number_of_days + days_in_advance + sell_window + 1:
      continue
    length = len(column)
    debug_buy_prices = []
    for i in range(0, length - number_of_days - days_in_advance - sell_window - 1):      
      dependent_variables.append(np.array(column[i:(i+number_of_days)]))
      window_buy_prices = column[(i+number_of_days+days_in_advance):(i+number_of_days+days_in_advance+buy_window)]
      buy_price = np.min(window_buy_prices)
      window_sell_prices = column[(i+number_of_days+days_in_advance+1):(i+number_of_days+days_in_advance+1+sell_window)]
      '''should really factor in the buy_window in the above line
      going to ignore sell window too and set that to 1'''
      sell_price = np.max(window_sell_prices)
      last_sell_price = window_sell_prices[-1]
      actual_buy_prices.append([buy_price])
      debug_buy_prices.append(buy_price)
      actual_sell_prices.append([sell_price])
      last_sell_prices.append([last_sell_price])  
    dict = {}
    dict['prices'] = column
    dict['actual_buy_prices'] = np.concatenate([np.repeat(np.nan, number_of_days + days_in_advance), debug_buy_prices, np.repeat(np.nan, sell_window + 1)])
    if len(debug_buy_prices) == 0:
      continue
    df = pd.DataFrame(data=dict)    
    df.to_csv('debugging/'+company+'.txt', index=False, sep='\t', na_rep = 'NaN', header=True)
  return actual_buy_prices, actual_sell_prices, last_sell_prices, dependent_variables
    
def produce_actual_matrix(data, companies, days_in_advance=1):
  actual_matrix = []
  for company in companies:
    column = data[company]
    column = pd.Series.dropna(column)
    if len(column) == 0:
      continue
    actual_matrix.append(np.array(column))
  return actual_matrix

def turn_vector_into_arrays(vector, days_in_advance = 1, number_of_days = 30, buy_window = 1, sell_window = 5):
  vector = np.array(vector)
  actual_buy_prices = []
  actual_sell_prices = []
  last_sell_prices = []
  dependent_variables = []
  length = len(vector)
  column = vector
  for i in range(0, length - number_of_days - days_in_advance - sell_window - 1):
    dependent_variables.append(np.array(column[i:(i+number_of_days)]))
    window_buy_prices = column[(i+number_of_days+days_in_advance):(i+number_of_days+days_in_advance+buy_window)]
    buy_price = np.min(window_buy_prices)
    window_sell_prices = column[(i+number_of_days+days_in_advance+1):(i+number_of_days+days_in_advance+1+sell_window)]
    sell_price = np.max(window_sell_prices)
    last_sell_price = window_sell_prices[-1]
    actual_buy_prices.append([buy_price])
    actual_sell_prices.append([sell_price])
    last_sell_prices.append([last_sell_price])      
  return actual_buy_prices, actual_sell_prices, last_sell_prices, dependent_variables
    

def is_a_number(x):
  try:
    y = x * 2
    if not np.isnan(y):
      return True
  except:
    return False

def get_prices_and_earnings(companies, lag = 1):
  prices_result = []
  earnings_result = []
  current_prices_result = []
  dates = []
  for company in companies:
    dates_to_append = []
    try:
      table = pd.read_csv('Earnings/' + company + '_quarterly_financial_data.csv')
    except:
      continue
    earnings = np.array(table['EPS basic'])
    prices = np.array(table['Price'])
    dates_to_append = table['Quarter end'].tolist()
    dates_to_append.reverse()
    earnings = np.flip(earnings)
    prices = np.flip(prices)
    for i in range(len(earnings)):
      try:
        earnings[i] = float(earnings[i])
      except:
        pass
      try:
        prices[i] = float(prices[i])
      except:
        pass
      if not is_a_number(earnings[i]):
        if i == 0:
          earnings[i] = np.nan
          prices[i] = np.nan
        else:
          earnings[i] = earnings[i-1]
          if np.isnan(earnings[i]):
            prices[i] = np.nan
      if not is_a_number(prices[i]):
        if i == 0:
          prices[i] = np.nan
          earnings[i] = np.nan
        else:
          prices[i] = prices[i-1]
          if np.isnan(prices[i]):
            earnings[i] = np.nan  
    for i in range(len(prices)):
      if np.isnan(prices[i]):
        dates_to_append[i] = 'REMOVE'
    dates_to_append = [x for x in dates_to_append if not x == 'REMOVE']  
    prices = [x for x in prices if not np.isnan(x)]
    earnings = [x for x in earnings if not np.isnan(x)]
    current_prices = prices
    current_prices = current_prices[0: (len(current_prices)-lag)]
    future_prices = prices[lag:len(prices)]
    dates_to_append = dates_to_append[lag:len(dates_to_append)]
    earnings = earnings[0:(len(earnings)-lag)]
    prices_result = np.concatenate((prices_result, future_prices))
    earnings_result = np.concatenate((earnings_result, earnings))
    current_prices_result = np.concatenate((current_prices_result, current_prices))
  prices_result = np.array([float(x) for x in prices_result], dtype = np.float32)
  earnings_result = np.array([float(x) for x in earnings_result], dtype = np.float32)
  current_prices_result = np.array([float(x) for x in current_prices_result], dtype = np.float32)
  dates.append(dates_to_append)
  return prices_result, earnings_result, current_prices_result, dates

def get_todays_predicted_price_and_current_price(company, model):
  try:
    table = pd.read_csv('Earnings/' + company + '_quarterly_financial_data.csv')
  except:
    return None
  earnings = np.array(table['EPS basic'])
  prices = np.array(table['Price'])
  earnings = np.flip(earnings)
  prices = np.flip(prices)
  last_current_price = prices[-1]
  last_earnings = earnings[-1]
  current_price = get_todays_price(company)
  todays_predicted_price = model.predict([current_price], [last_current_price], [last_earnings])[0]
  return todays_predicted_price, current_price

def get_next_predicted_price(company, model):
  current_price, earnings = get_todays_price_and_earnings(company)
  predicted_price = model.predict([0], [current_price], [earnings])[0]
  return predicted_price


def process_returns_array(returns_array):
  total = 0.0
  result = []
  for i in range(len(returns_array)):
    current = returns_array[i]
    if not np.isnan(current) and not current == 0.0:
      total = total + current - 1
    result.append(total)
  result = np.array(result)
  result = 100 * result
  return result

def save_dataframe_in_demo_directory(companies, test_data, predicted_sell_prices, returns, number_of_days, days_in_advance, sell_window, test_actual_buy_prices):
  predicted_sell_prices = np.reshape(predicted_sell_prices, -1)
  returns = np.reshape(returns, -1)
  i = 0
  for company in companies:
    column = test_data[company]
    column = pd.Series.dropna(column)
    number_to_add = len(column) - number_of_days - days_in_advance - sell_window - 1
    if number_to_add < 0:
      continue
    dict = {}
    dict['prices'] = column
    predicted_sell_prices_array = predicted_sell_prices[i:(i+number_to_add)]
    returns_array = returns[i:(i+number_to_add)]
    predicted_sell_prices_array = np.concatenate([np.repeat(np.nan, number_of_days + days_in_advance), predicted_sell_prices_array,  np.repeat(np.nan, sell_window + 1)])
    returns_array = np.concatenate([np.repeat(np.nan, number_of_days + days_in_advance), returns_array, np.repeat(np.nan, sell_window + 1)])
    returns_array = process_returns_array(returns_array)
    buy_prices = test_actual_buy_prices[i:(i+number_to_add)]
    buy_prices = np.reshape(buy_prices, -1)
    buy_prices = np.concatenate([np.repeat(np.nan, number_of_days + days_in_advance), buy_prices, np.repeat(np.nan, sell_window + 1)])
    dict['predicted_sell_prices'] = predicted_sell_prices_array
    dict['returns'] = returns_array
    df = pd.DataFrame(data=dict)

    '''
    also now want to add style to the dataframe; different type of point if df['prices']<df['predicted_sell_prices']
    also different type of point if it is days_in_advance + sell_window along from a buy - may not do this for the moment?
    
    '''


    df.to_csv('demo/'+company+'.csv', index='dates', sep='\t', na_rep = 'NaN', header=True)
    i = i + number_to_add

def test1():
  if not 'demo' in os.listdir('.'):
    os.mkdir('demo')
  companies = get_companies()[0:120]
  to_exclude = ['ICT', 'ABK', 'BHL', 'BMC', 'CFL', 'AMO', 'ATN', 'ACS', 'CBJ']
  companies = [x for x in companies if not x in to_exclude]
  data  = get_data(companies, start="2017-01-01", end="2018-12-31", use_stored=False)
  number_of_days = 30
  days_in_advance = 0 
  sell_window = 1
  actual_buy_prices, actual_sell_prices, last_sell_prices, dependent_variables = make_arrays(data, companies, number_of_days=number_of_days, days_in_advance = days_in_advance, sell_window = sell_window) 
  if not 'weights.npy' in os.listdir('.'):
    model = NewModel_latest(actual_buy_prices, actual_sell_prices, last_sell_prices, dependent_variables, positive = False, load_previous=False, weights=None)
    model.train()
    weights = model.sess.run(model.W_buy)
    bias = model.sess.run(model.bias)
    np.save('weights.npy', weights)
    np.save('bias.npy', bias)
  else:
    print('Loading previous weights')
    weights = np.load('weights.npy')
    bias = np.load('bias.npy')
    model = NewModel_latest(actual_buy_prices, actual_sell_prices, last_sell_prices, dependent_variables, positive = False, load_previous=True, weights=weights, bias=bias)
  companies = get_companies()[0:120]
#   companies = companies + ['FB']
  to_exclude = ['ICT', 'ABK', 'BHL', 'BMC', 'CFL', 'AMO', 'ATN', 'ACS', 'CBJ']
  companies = [x for x in companies if not x in to_exclude]  
  test_data  = get_data(companies, start="2019-01-01", end="2019-12-31", use_stored=False)
  test_actual_buy_prices, test_actual_sell_prices, test_last_sell_prices, test_dependent_variables = make_arrays(test_data, companies, number_of_days=number_of_days, days_in_advance = days_in_advance, sell_window=sell_window) 
  predicted_sell_prices, returns = model.test(test_actual_buy_prices, test_actual_sell_prices, test_last_sell_prices, test_dependent_variables)
  save_dataframe_in_demo_directory(companies, test_data, predicted_sell_prices, returns, number_of_days, days_in_advance, sell_window, test_actual_buy_prices)
  

  
test1()






