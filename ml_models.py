import base64, io
from math import inf
from statistics import mode
from scipy.spatial import distance
from numpy import argmax
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier



def decision_tree_classifier(max_depth, X_train, y_train, X_test, y_test):
    # create a simple model and fit it with training dataset
    clf = DecisionTreeClassifier(max_depth=max_depth).fit(X_train, y_train)
    
    # predict the fitted model
    y_pred = clf.predict(X_test)

    # create a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    # return predction accuracy, figure object and predicted classes
    return round(clf.score(X_test, y_test), 3), disp, y_pred


def logistic_regression(max_iter, X_train, y_train, X_test, y_test, penalty='none', solver='saga'):
    # create a simple model and fit it with training dataset
    log_reg = LogisticRegression(penalty=penalty, solver=solver, max_iter=max_iter).fit(X_train, y_train)
    
    # predict the fitted model
    y_pred = log_reg.predict(X_test)

    # create a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    # return predction accuracy, figure object and predicted classes
    return round(log_reg.score(X_test, y_test), 3), disp, y_pred


def knn(k, X_train, y_train, X_test, y_test):
    min_distances = []
    y_pred = []
    for point_test in range(X_test.shape[0]):
        min_distances.append([(inf, 0) for _ in range(k)])  # tuple (distance, class)
        
        # loop to count the smallest distances between test point and surrounding points 
        for point_train in range(X_train.shape[0]):
            min_distances[point_test].sort()
            dist = distance.euclidean(X_test[point_test], X_train[point_train])          
            if dist < min_distances[point_test][-1][0]:
                min_distances[point_test][-1] = dist, y_train[point_train]
        
        # predicted class is the class with most occurences
        y_pred.append(mode([min_distances[point_test][i][1] for i in range(k)]))

    # save an object of confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    # return predction accuracy, figure object and predicted classes
    return round(accuracy_score(y_test, y_pred), 3), disp, y_pred


def create_nn_model(hidden_layer1_neurons, hidden_layer2_neurons, dropout, learning_rate, epochs, X_train, y_train, X_test, y_test):
    # create a neural network model with user's provided parameters for number of neurons, dropout value, learning rate and number of epochs 
    model = keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(54,)),
        keras.layers.Dense(hidden_layer1_neurons, activation='relu'),
        keras.layers.Dense(hidden_layer2_neurons, activation='relu'),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(7)
    ])
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs)

    # predict the fitted model and choose the maximum value in a logit for each sample
    y_pred = argmax(model.predict(X_test), axis=1)

    # save an object of confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    # return predction accuracy, figure object and predicted classes
    return round(model.evaluate(X_test, y_test)[1], 3), disp, y_pred


def buffer_plot(disp):
    # generate the confusion matrix plot to save it to a buffer
    buffer = io.BytesIO()
    disp.plot(include_values=True, cmap='Blues')
    plt.savefig(buffer, format='png')
    plt.close()

    # convert the buffer to a base64-encoded string and return it
    buffer.seek(0)
    confusion_matrix = base64.b64encode(buffer.read()).decode()
    buffer.close()

    return confusion_matrix



# model for evaluations only!
def create_nn_model_eval(hidden_layer1_neurons, hidden_layer2_neurons, dropout, learning_rate):
    # create a neural network model with user's provided parameters for number of neurons, dropout value, learning rate and number of epochs 
    model = keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(54,)),
        keras.layers.Dense(hidden_layer1_neurons, activation='relu'),
        keras.layers.Dense(hidden_layer2_neurons, activation='relu'),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(7)
    ])
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    return model