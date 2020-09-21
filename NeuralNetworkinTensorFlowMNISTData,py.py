# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 18:35:10 2020

@author: sefir
"""

#Tensorflow: Net input Z, X in 1-D data set with W and Bias Z=W*X+B
import tensorflow as tf
## create a graph
g = tf.Graph()
with g.as_default():
    x = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None))
    w = tf.Variable(2.0, name='weight')
    b = tf.Variable(0.7, name='bias')
    z = w*x + b
    init = tf.compat.v1.global_variables_initializer()
    
with tf.compat.v1.Session(graph=g) as sess:
    ## initialize w and b
    sess.run(init)
    ## evaluate z:
    for t in [1.0, 0.6, -1.8]:
        print('x=%4.1f --> z=%4.1f'% (t, sess.run(z, feed_dict={x:t})))
        print('')
with tf.compat.v1.Session(graph=g) as sess:
    sess.run(init)
    print(sess.run(z, feed_dict={x:[1., 2., 3.]}))

#Running a Neural Network using Tensorflow
import tensorflow as tf
# Load and prepare the MNIST dataset. Convert the samples from integers to floating-point numbers:
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
import matplotlib.pyplot as plt
image_index = 7777 # You may select anything up to 60,000
print(y_train[image_index]) # The label is 8
plt.imshow(x_train[image_index], cmap='Greys')
x_train, x_test = x_train / 255.0, x_test / 255.0
# Building the model structure
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])
# For each example the model returns a vector of "logits" or "log-odds" scores, one for each class.
predictions = model(x_train[:1]).numpy()
predictions
#The tf.nn.softmax function converts these logits to "probabilities" for each class:
tf.nn.softmax(predictions).numpy()
# and a True index and returns a scalar loss for each example.
# The losses.SparseCategoricalCrossentropy loss takes a vector of logits 
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# This loss is equal to the negative log probability of the true class: 
# It is zero if the model is sure of the correct class.
# This untrained model gives probabilities close to random
# (1/10 for each class), so the initial loss should be 
#close to -tf.log(1/10) ~= 2.3.
loss_fn(y_train[:1], predictions).numpy()
# Compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Training the model:
# The Model.fit method adjusts the model parameters to minimize the loss:
history = model.fit(x_train, y_train, epochs=5)
# Evaluating the model
# The Model.evaluate method checks the models performance, usually on a "Validation-set" or "Test-set".
model.evaluate(x_test, y_test, verbose=2)
# The image classifier is now trained to ~98% accuracy on this dataset. 
# If you want your model to return a probability, 
# you can wrap the trained model, and attach the softmax to it:
probability_model = tf.keras.Sequential([model,
  tf.keras.layers.Softmax()
])
print(probability_model(x_test[:5]))

import matplotlib.pyplot as plot
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

image_index = 4444
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
pred = model.predict(x_test[image_index].reshape(1, 28, 28))
print(pred.argmax())


