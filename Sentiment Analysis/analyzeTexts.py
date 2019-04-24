import json
from collections import Counter
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk
import re
from random import shuffle
import matplotlib.pyplot as plt
import csv
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
nltk.download('stopwords')


# Function to load data from Amazon cell phone & accessories category
def load_amazon(subset):
    train_dict = {}
    document = 1
    reviews = []
    sentiments = []
    counter = 0
    if subset == 'cell phones':
        print('Loading cell data...')
        with open('data/amazon/data.json', 'r') as f:
            for line in f:
                train_dict[document] = json.loads(line)
                document += 1
    elif subset == 'movies':
        print('Loading movie data...')
        with open('data/movies/amazonMovies.json', 'r') as f:
            for line in f:
                train_dict[document] = json.loads(line)
                document += 1
                counter += 1
                # Parse 800000 reviews
                if counter == 800000:
                    break
    # Iterate over nested python dictionary and save review text and sentiments
    # train_dict contains one id for each review (starts from 1 and goes up to #reviews) and all the json data
    # from the file # for each id.
    for docID in train_dict:
        reviews.append(train_dict[docID]['reviewText'])
        sentiments.append(train_dict[docID]['overall'])
    # Release some memory
    del train_dict
    return reviews, sentiments


def load_twitter():
    corpus = []
    sentiments = []
    # Read train set file and skip first row
    print('Loading twitter data...')
    with open('data/twitter/train_set.csv', 'r', encoding='utf8') as trainfile:
        reader = csv.reader(trainfile, delimiter=',')
        next(reader)
        for row in reader:
            # Append tweet to corpus list
            corpus.append(row[-1])
            # Append sentiment to sentiments list
            # 0 is negative and 4 is positive => mapped to 0 and 1
            # This is a binary problem.
            if row[0] == '0':
                sentiments.append(0)
            elif row[0] == '4':
                sentiments.append(1)

    # First, we need to shuffle the lists in the same order, because in the original data they are not shuffled.
    zipped_list = list(zip(corpus, sentiments))
    shuffle(zipped_list)
    corpus, sentiments = zip(*zipped_list)
    # For memory reasons, we will use 50% of the available corpus.
    cutoff_index = int(len(corpus) / 2)
    corpus = corpus[:cutoff_index]
    sentiments = sentiments[:cutoff_index]
    return corpus, sentiments


# Function to generate wordcloud from raw data
# Dataset is cell phones/twitter/movies
def generate_cloud(data, dataset):
    # Add some stopwords depending on the dataset picked
    print('Generating wordcloud...')
    if dataset == 'cell phones':
        phone_stopwords = ['phone', 'phone case', 'iPhone', 'micro USB', 'USB', 'cover', 'charger', 'cell phone',
                           'case']
        stop_words = set(phone_stopwords + list(stopwords.words('english')))
    elif dataset == 'twitter':
        twitter_stopwords = ['hey', 'um', 'umm', 'http', 'lol']
        stop_words = set(twitter_stopwords + list(stopwords.words('english')))
    elif dataset == 'movies':
        movie_stopwords = ['film', 'director', 'script', 'scenewriter', 'scene','one','movie']
        stop_words = set(movie_stopwords + list(stopwords.words('english')))
    corpus = " ".join(text for text in data)
    wc = WordCloud(background_color="white", stopwords=stop_words).generate(corpus)
    plt.title('Most common words')
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# Function to clean the twitter corpus. Twitter data is very "dirty", and contains a lot of irrelevant info
# which can affect accuracy significantly
def preprocess(tweets):
    # Remove http links
    clean_corpus = []
    for tweet in tweets:
        result = re.sub(r"http\S+", "", tweet)
        clean_corpus.append(result)
    return clean_corpus


def main():
    # AMAZON CELL PHONES
    amazon_data = load_amazon(subset='cell phones')
    amazon_reviews = amazon_data[0]
    amazon_sentiments = amazon_data[1]
    del amazon_data
    print('Class distribution for Amazon cell phone data:', Counter(amazon_sentiments))
    # Gather all highly positive and really negative reviews in two lists to analyze with wordclouds
    amazon_positive = []
    amazon_negative = []
    for review, sentiment in zip(amazon_reviews, amazon_sentiments):
        if sentiment == 5:
            amazon_positive.append(review)
        elif sentiment == 1:
            amazon_negative.append(review)
    # Generate wordcloud for amazon really positive reviews
    generate_cloud(amazon_positive, dataset='cell phones')

    # TWITTER
    twitter_data = load_twitter()
    twitter_tweets = twitter_data[0]
    twitter_tweets = preprocess(twitter_tweets)
    twitter_sentiments = twitter_data[1]
    del twitter_data
    print('Class distribution for Twitter data:', Counter(twitter_sentiments))
    # Gather all highly positive and really negative reviews in two lists to analyze with wordclouds
    twitter_positive = []
    twitter_negative = []
    for tweet, sentiment in zip(twitter_tweets, twitter_sentiments):
        if sentiment == 5:
            twitter_positive.append(tweet)
        elif sentiment == 1:
            twitter_negative.append(tweet)
    # Generate wordcloud for negative tweets
    generate_cloud(twitter_negative, dataset='twitter')

    # MOVIES
    movie_data = load_amazon(subset='movies')
    movie_reviews = movie_data[0]
    movie_sentiments = movie_data[1]
    del movie_data
    print('Class distribution for movie data:', Counter(movie_sentiments))
    # Gather all highly positive and really negative reviews in two lists to analyze with wordclouds
    movie_positive = []
    movie_negative = []
    for review, sentiment in zip(movie_reviews, movie_sentiments):
        if sentiment == 5:
            movie_positive.append(review)
        elif sentiment == 1:
            movie_negative.append(review)
    # Generate wordcloud for really negative movie reviews
    generate_cloud(movie_negative, dataset='movies')


main()
