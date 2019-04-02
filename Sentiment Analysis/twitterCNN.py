import csv
from collections import Counter
import re
from random import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np

# Parameters for keras
# Max words is how many top words to consider (term frequency is used)
top_words = 5000
dropout_perc = 0.3
maxlen = 400
embedding_size = 40
batch_size = 64
filters = 250
kernel_size = 3
stride = 1
epochs = 15
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
        # 0 is negativeand 4 is positive => mapped to 0 and 1
        # This is a binary problem.
        if row[0] == '0':
            sentiments.append(0)
        elif row[0] == '4':
            sentiments.append(1)

# First, we need to shuffle the lists in the same order, because in the original data they are not shuffled.
zippedList = list(zip(corpus, sentiments))
shuffle(zippedList)
corpus, sentiments = zip(*zippedList)
# For memory reasons, we will use 50% of the available corpus. 80% will be training data, 10% validation and 10% test
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
# Train a keras tokenizer and transform tweets
print('Training tokenizer...')
tokenizer = keras.preprocessing.text.Tokenizer(num_words=top_words)
tokenizer.fit_on_texts(x_train)
# More preprocessing. Each tweet is transformed to a series of integers where each integer represents the associated
# word's frequency in the entire corpus.
print('Transforming texts to vectors...')
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
x_val = tokenizer.texts_to_sequences(x_val)
print('Sample vectorized text', x_train[0])
vocabulary_size = max(np.amax(x_train))
print('Vocabulary size: ', vocabulary_size)

# Pad each sequence to a fixed size
print('Padding sequences to a fixed size of ', maxlen)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)
print('Train dataset shape is ', x_train.shape)
# Built Tensorflow-Keras CNN Model. First, use an Embedding layer to map each integer to a fixed size vector for
# better representation. Then, run a Convolutional1D filter over the vector and use max pooling.
# Dropout is used for regularization
model = keras.Sequential()
# Stop training early if that is desired.
callbacks = [
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor='val_loss',
        min_delta=1e-2,
        # Also wait for at least 3 epochs before deciding whether to stop training
        patience=2,
        verbose=1)
]
model.add(keras.layers.Embedding(top_words, embedding_size, input_length=maxlen))
model.add(keras.layers.Dropout(dropout_perc))
model.add(keras.layers.Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=stride))
model.add(keras.layers.GlobalMaxPooling1D())
# Add a hidden layer
model.add(keras.layers.Dense(maxlen, activation='relu'))
model.add(keras.layers.Dropout(dropout_perc))
model.add(keras.layers.Dense(1))
model.add(keras.layers.Activation('sigmoid'))

# Binary classification problem: use binary_crossentropy, sigmoid activation and adam/rmsprop
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Can stop training early by removing the # next to the callbacks argument.
history = model.fit(x_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    # callbacks=callbacks,
                    validation_data=(x_val, y_val),
                    verbose=1)

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
with open('statistics/twitter/twitterCNN.csv', 'a', encoding='utf8', newline='') as outfile:
    writer = csv.writer(outfile, delimiter=',')
    # Record: Vocabulary size, review length, embedding size, batch_size, filters, kernel_size, stride, epochs,
    # dropout_perc, test_acc
    writer.writerow([top_words, maxlen, embedding_size, batch_size, filters, kernel_size, stride, epochs, dropout_perc,
                     results[1]])
