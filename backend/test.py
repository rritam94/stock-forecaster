import requests
import time

def calculate_sentiment():
    global t
    t = time.time()
    symbol = 'MSFT'
    news = requests.get('https://www.benzinga.com/quote/' + symbol)
    parse_webpage(news.text)

def parse_webpage(news: str):
    headlines = []
    headline_indices = []
    headline_index = 0

    date_indices = []
    date_index = 0
    idx = 0

    while True:
        headline_index = news.find('noopener noreferrer', headline_index + 1)
        date_index = news.find('"text-gray-500">', date_index + 1)

        if headline_index == -1:
            break

        if date_index == -1:
            break

        headline_indices.append(headline_index)
        date_indices.append(date_index)

    for index in headline_indices:
        headline = ''
        curr_index = index

        # gets the date element only (3 days ago, 6/21/2023, etc)
        if news[date_indices[idx] + 16 : date_indices[idx] + 30].find('3 day') != -1:
            print(news[date_indices[idx] + 16 : date_indices[idx] + 30])
            break

        while (news[curr_index] != '>'):
            curr_index = curr_index + 1

        while (news[curr_index] != '<'):
            headline = headline + news[curr_index]
            curr_index = curr_index + 1

        headlines.append(headline)
        idx = idx + 1

calculate_sentiment()