FROM tensorflow/tensorflow:1.13.1-py3
RUN pip install yfinance flask flask_restful wtforms
RUN mkdir /stocks
WORKDIR /stocks
