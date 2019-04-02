import csv
from collections import Counter
import re
from random import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Parameters for keras
# Max words is how many top words to consider (term frequency is used)
epochs = 50
batch_size = 32
hidden_layer_size = 200
dropout_perc = 0.5
max_words = 10000
n_components = 200
# One list for the tweet corpus, one list for each corresponding sentiment
corpus = []
sentiments = []

# Read train set file and skip first row
print('Loading data...')
with open('data/twitter/train_set.csv', 'r', encoding='utf8') as trainfile:
    reader = csv.reader(trainfile, delimiter=',')
    headers = next(reader)
    for row in reader:
        # Append tweet to corpus list
        corpus.append(row[-1])
        # Append sentiment to sentiments list
        # 0 is negative, 2 is neutral and 4 is positive => mapped to 0, 1 and 2
        # This is a multiclass problem.
        if row[0] == '0':
            sentiments.append(0)
        elif row[0] == '4':
            sentiments.append(1)

# First, we need to shuffle the lists in the same order, because in the original data they are not shuffled.
zippedList = list(zip(corpus, sentiments))
shuffle(zippedList)
corpus, sentiments = zip(*zippedList)
# For memory and speed reasons, we will use 50% of the available corpus. 80% will be training data, 10% validation and
# 10% test.
cutoffIndex = int(len(corpus) / 2)
corpus = corpus[:cutoffIndex]
sentiments = sentiments[:cutoffIndex]
# Check class balance
print(Counter(sentiments))


# Function to clean the corpus. Twitter data is very "dirty", and contains a lot of irrelevant info which can affect
# accuracy signifcantly
def preprocess(texts):
    # Remove http links
    cleanCorpus = []
    for tweet in texts:
        result = re.sub(r"http\S+", "", tweet)
        cleanCorpus.append(result)
    return cleanCorpus


corpus = preprocess(corpus)
# print(corpus)
# Since we have 800.000 patterns, split is 640.000/72.000/72.000. Class distribution is balanced and data was randomly
# shuffled, so we can safely sample the data in sequence.
x_train = corpus[:640000]
y_train = sentiments[:640000]
x_val = corpus[640000:720000]
y_val = sentiments[640000:720000]
x_test = corpus[720000:]
y_test = sentiments[720000:]
# Release some memory
del corpus, sentiments
print('Vectorizing data...')
vectorizer = TfidfVectorizer(strip_accents='unicode', max_features=2500)
x_train = vectorizer.fit_transform(x_train)
print(vectorizer.vocabulary_)
x_val = vectorizer.transform(x_val)
x_test = vectorizer.transform(x_test)
print('Training corpus shape after tf idf is: ', x_train.shape)
# Perform dimensionality reduction with help from sklearn and keep the most relevant 200 features
# TruncatedSVD is used
print('Performing SVD...')
svd = TruncatedSVD(n_components=n_components)
x_train = svd.fit_transform(x_train)
x_val = svd.transform(x_val)
x_test = svd.transform(x_test)
print('Data shape after SVD is ', x_train.shape)
# Built Tensorflow-Keras MLP model
model = keras.Sequential()
# Can stop training early if that is desired.
callbacks = [
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor='val_loss',
        min_delta=1e-2,
        # Also wait for at least 3 epochs before deciding whether to stop training
        patience=3,
        verbose=1)
]
model.add(keras.layers.Dense(hidden_layer_size, input_shape=(n_components,), activation='relu'))
model.add(keras.layers.Dropout(dropout_perc))
model.add(keras.layers.Dense(1))
model.add(keras.layers.Activation('sigmoid'))

# Binary classification problem: use binary_crossentropy, sigmoid activation and adam/rmsprop.
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Remove the # next to callbacks for early stopping
history = model.fit(x_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    # callbacks=callbacks,
                    validation_data=(x_val, y_val),
                    verbose=1
                    )
results = model.evaluate(x_test, y_test, verbose=1)
print(results)
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Save some statistics to a csv
with open('statistics/twitter/twitterMLP.csv', 'a', encoding='utf8', newline='') as outfile:
    writer = csv.writer(outfile, delimiter=',')
    # Record: Vocabulary size, svd components, hidden layer size, batch_size, epochs, dropout_perc, test_acc
    writer.writerow([max_words, n_components, hidden_layer_size, batch_size, epochs, dropout_perc, results[1]])
