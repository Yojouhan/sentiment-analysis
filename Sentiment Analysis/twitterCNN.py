import csv
from collections import Counter
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
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
epochs = 10
# One list for the tweet corpus, one list for each corresponding sentiment
corpus = []
sentiments = []
counter = 0
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
        counter += 1
        # Check if we loaded the negative tweets, so as to skip 400000 rows to subsample original dataset
        if counter == 400000:
            for skip in range(400000):
                next(reader)

        # Stop when 800000 tweets are loaded
        if len(corpus) > 800000:
            break

# Check class balance
print(Counter(sentiments))


# Function to clean the corpus. Twitter data is very "dirty", and contains a lot of irrelevant info which can affect
# accuracy signifcantly
def preprocess(texts):
    # Remove http links
    clean_corpus = []
    for tweet in texts:
        result = re.sub(r"http\S+", "", tweet)
        clean_corpus.append(result)
    return clean_corpus


corpus = preprocess(corpus)
# print(corpus)
# Since we have 800.000 patterns, split is 640.000/160.000.
x_train, x_test, y_train, y_test = train_test_split(corpus, sentiments,
                                                    stratify=sentiments,
                                                    test_size=0.2)

print(Counter(y_train))

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

print('Sample vectorized text', x_train[0])
vocabulary_size = max(np.amax(x_train))
print('Vocabulary size: ', vocabulary_size)

# Pad each sequence to a fixed size
print('Padding sequences to a fixed size of ', maxlen)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

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
                    validation_split=0.1,
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
