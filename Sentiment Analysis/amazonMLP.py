import json
import csv
import os
import tensorflow as tf
from tensorflow import keras
from collections import Counter
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
# Parameters for keras
# Max words is how many top words to consider (term frequency is used)
epochs = 50
batch_size = 32
hidden_layer_size = 200
dropout_perc = 0.3
max_words = 5000
n_components = 200
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

# Release some memory as the dict is no longer needed
del train_dict
# Count class occurence
print(Counter(sentiments))
# Split into train and test data. We use a 80-20 split here.
# Also convert the sentiments list to a list of ints, to save memory
sentiments = np.array([int(i - 1) for i in sentiments])
# Split into train and test set
x_train = reviews[0:int(0.8 * len(reviews))]
y_train = sentiments[0:int(0.8 * len(sentiments))]
x_test = reviews[int(0.8 * len(reviews)):]
y_test = sentiments[int(0.8 * len(sentiments)):]
print('Train test size is', len(x_train), 'Test set size is ', len(x_test))
# Perform dimensionality reduction with help from sklearn and keep the most relevant 200 features
# TruncatedSVD is used
print('Performing SVD...')
vectorizer = TfidfVectorizer(max_features=max_words)
x_train = vectorizer.fit_transform(x_train)
print('Data shape before SVD is ', x_train.shape)
svd = TruncatedSVD(n_components=n_components)
x_train = svd.fit_transform(x_train)
# Also transform the test set
x_test = vectorizer.transform(x_test)
x_test = svd.transform(x_test)
# Number of classes (5)
num_classes = np.max(sentiments) + 1
print('Number of classes:', num_classes)
print(len(y_train))
# Convert to one hot encoding
y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]

# Built Tensorflow-Keras MLP model
model = keras.Sequential()
# Stop training early if that is desired
callbacks = [
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor='val_loss',
        min_delta=1e-2,
        # Also wait for at least 3 epochs before deciding whether to stop training
        patience=5,
        verbose=1)
]
# A simple fully connected architecture: input->dense layer->output
model.add(keras.layers.Dense(hidden_layer_size, input_shape=(n_components,), activation='relu'))
model.add(keras.layers.Dropout(dropout_perc))
model.add(keras.layers.Dense(num_classes))
model.add(keras.layers.Activation('softmax'))
# Save model architecture to file
keras.utils.plot_model(model, to_file='modelMLP.png', show_shapes=True, show_layer_names=True)
# Multiclass problem-> should use categorical crossentropy, and adam or rmsprop preferably.
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# To stop training early, add a # before the callbacks argument.
history = model.fit(x_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
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
with open('statistics/amazon/amazonMLP.csv', 'a', encoding='utf8', newline='') as outfile:
    writer = csv.writer(outfile, delimiter=',')
    # Record: Vocabulary size, svd components, hidden layer size, batch_size, epochs, dropout_perc, test_acc
    writer.writerow([max_words, n_components, hidden_layer_size, batch_size, epochs, dropout_perc, results[1]])
