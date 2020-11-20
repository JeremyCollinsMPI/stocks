
import requests
import re


r = requests.get('http://www.stockpup.com/data/')
thing = r.text.split(' ')
csv_files = [x.split('/')[-1].split('csv')[0]+'csv' for x in thing if '.csv' in x and not 'CSV' in x]
with requests.Session() as s:
  for file in csv_files:
    url = 'http://www.stockpup.com/data/' + file
    download = s.get(url)
    decoded_content = download.content
    filename = 'Earnings/' + file
    open(filename, 'wb').write(decoded_content)
