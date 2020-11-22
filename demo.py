from flask import Flask, render_template, request, redirect
import json
from time import sleep
import pandas as pd
from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import os


app = Flask(__name__)

def produce_sorted_text():
  dir = os.listdir('demo')
  sorted_text = open('sorted.txt','w')
  dir.remove('.DS_Store')
  for file in dir:
    print(file)
    df = pd.read_csv('demo/' + file, header=0, sep='\t')
    print(df)
    sorted_text.write(file.replace('.csv','') + '\t' + str(round(df['returns'].tolist()[-1], 2)) + '\n')

def get_sorted_stocks():
  if not 'sorted.txt' in os.listdir('.'):
    produce_sorted_text()
  file = open('sorted.txt', 'r').readlines()
  to_return = [x.split('\t') for x in file]
  to_return = sorted(to_return, key = lambda x:float(x[1]), reverse=True)
  return to_return 
        
@app.route('/company', methods=['GET', 'POST'])
def s(sorted_stocks = None, company = None):

    if company == None:
        company = request.args.get('name')
    print('****')
    print(company)
    company = company.upper()
    if company + '.csv' in os.listdir('/bbc'):
        template = 'demo_with_finbert.html'
    else:
        template = 'demo.html'
    sorted_stocks = get_sorted_stocks()
    
    table = pd.read_csv('demo/'+company+'.csv', sep='\t', na_values='NaN')
    array = []
    columns = table.columns.to_list()
    array.append(columns)
    for index, row in table.iterrows():
      array.append(row.to_list())
    dict = {'array': array}
    finbert_table = None
    if company + '.csv' in os.listdir('/bbc'):
        finbert_table = pd.read_csv('/bbc/demo/' + company + '.csv', sep='\t')
        print(finbert_table)
        finbert_array = []
#         finbert_columns = finbert_table.columns.to_list()
#         finbert_array.append(finbert_columns)
        for index, row in finbert_table.iterrows():
            finbert_array.append(row.to_list())
        finbert_table = finbert_array
        print(finbert_table)

    return render_template(template, moose = company.upper(), dict = dict, sorted_stocks=sorted_stocks, finbert_table=finbert_table)



class ReusableForm(Form):
    name = TextField('Search term:', validators=[validators.required()])
    @app.route("/home", methods=['GET', 'POST'])
    def hello():
        sorted_stocks = get_sorted_stocks()
#         company = sorted_stocks[0][0]
        company = 'FB'
        company = company.upper()
        table = pd.read_csv('demo/'+company+'.csv', sep='\t')
        array = []
        columns = table.columns.to_list()
        array.append(columns)
        for index, row in table.iterrows():
          array.append(row.to_list())
        dict = {'array': array}

        form = ReusableForm(request.form)
    
        print(form.errors)
        if request.method == 'POST':
            company = request.form['name']
            return s(sorted_stocks, company=company)
        finbert_table = pd.read_csv('/bbc/demo/FB.csv', sep='\t', header=None)
        finbert_array = []
#         finbert_columns = finbert_table.columns.to_list()
#         finbert_array.append(finbert_columns)
        for index, row in finbert_table.iterrows():
          finbert_array.append(row.to_list())
        finbert_table = finbert_array

        return render_template('demo_with_finbert.html', moose = company.upper(), dict = dict, sorted_stocks = sorted_stocks, finbert_table=finbert_table)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
    
'''
plan for demo
want to have a point where something is bought and another where it is sold
need to research how to plot points on a google graph
could use this
https://developers.google.com/chart/interactive/docs/points
see 'Customizing individual points'
'''