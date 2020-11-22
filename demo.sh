docker run -d --rm -v $PWD:/stocks -v ~/bbc:/bbc --name stocks -p 5000:5000 jeremycollinsmpi/stocks python demo.py
