import json
import tensorflow as tf
from tensorflow import keras
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import csv

# Network and preprocessing parameters: Number of most common words to keep, max review size, embedding vector size...
top_words = 5000
maxlen = 400
embedding_size = 50
batch_size = 32
filters = 250
kernel_size = 3
stride = 1
epochs = 20

# A dictionary that holds all review data, and a counter for each key. Also create a list that holds review text and
# a list that contains the sentiments. They'll then be processed in parallel.
train_dict = {}
reviews = []
sentiments = []
document = 1
# Parse json file into a dictionary, line by line.
print('Loading data...')
with open('data/amazon/data.json', 'r') as f:
    for line in f:
        train_dict[document] = json.loads(line)
        document += 1

# Iterate over nested python dictionary and save review text and sentiments
# train_dict contains one id for each review (starts from 1 and goes up to #reviews) and all the json data from the file
# for each id.
for docID in train_dict:
    reviews.append(train_dict[docID]['reviewText'])
    sentiments.append(train_dict[docID]['overall'])

# Release some memory as the dict is no longer needed. Can be helpful in machines with low RAM.
del train_dict
# Count class occurrence
print(Counter(sentiments))
# Split into train and test data. We use a 80-20 split here.
# Also convert the sentiments list to a list of ints, to save memory.
sentiments = np.array([int(i - 1) for i in sentiments])
x_train = reviews[0:int(0.8 * len(reviews))]
y_train = sentiments[0:int(0.8 * len(sentiments))]
x_test = reviews[int(0.8 * len(reviews)):]
y_test = sentiments[int(0.8 * len(sentiments)):]
print('Train test size is', len(x_train), 'Test set size is ', len(x_test))
# Number of classes (5)
num_classes = np.max(sentiments) + 1
print('Number of classes:', num_classes)
del sentiments, reviews

# Train a keras tokenizer and transform reviews
print('Training tokenizer...')
tokenizer = keras.preprocessing.text.Tokenizer(num_words=top_words)
tokenizer.fit_on_texts(x_train)
# More preprocessing. Each review is transformed to a series of integers where each integer represents the associated
# word's frequency in the entire corpus.
print('Transforming texts to vectors...')
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
print('Sample vectorized text', x_train[0])
vocabulary_size = max(np.amax(x_train))
print('Vocabulary size: ', vocabulary_size)

# Convert to one hot encoding for use with categorical crossentropy.
y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]

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
        patience=3,
        verbose=1)
]
model.add(keras.layers.Embedding(top_words, embedding_size, input_length=maxlen))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=stride))
model.add(keras.layers.GlobalMaxPooling1D())
# Add a hidden layer
model.add(keras.layers.Dense(maxlen, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(num_classes))
model.add(keras.layers.Activation('softmax'))

# Multiclass problem-> should use categorical crossentropy, and adam or rmsprop preferably.
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# Can stop training early by removing the # in the callbacks comment.
history = model.fit(x_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    # callbacks=callbacks,
                    validation_split=0.2,
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
with open('statistics/amazon/amazonCNN.csv', 'a', encoding='utf8', newline='') as outfile:
    writer = csv.writer(outfile, delimiter=',')
    # Record: Vocabulary size, review length, embedding size, batch_size, filters, kernel_size, stride, epochs, test_acc
    writer.writerow([top_words, maxlen, embedding_size, batch_size, filters, kernel_size, stride, epochs,
                     results[1]])
