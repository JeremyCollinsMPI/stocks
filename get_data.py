import yfinance as yf
#example



def get_data(companies, start, end, use_stored=False):
  if use_stored:
    pass
  else:
    data = yf.download(companies, start=start, end=end)
    print('0000asdfkih')
    print(data)
    print(data.loc[:,'Open'])
    return data.loc[:,'Open']

def get_todays_price(company):
  try:
    return yf.Ticker(company).info['regularMarketPreviousClose']['raw']
  except:
    return yf.Ticker(company).info['regularMarketPreviousClose']

def get_todays_price_and_earnings(company):
  data = yf.Ticker(company)
  try:
    earnings = data.info['forwardEps']['raw']
  except:
    earnings = data.info['forwardEps']
  try:
    price =  data.info['regularMarketPreviousClose']['raw']
  except:
    price = data.info['regularMarketPreviousClose']
  return price, earnings