from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from ml_models import decision_tree_classifier, logistic_regression, create_nn_model_eval, knn

data = read_csv('data/covtype.data')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)

# change labels from 1-7 to 0-6 
y_train -= 1
y_test -= 1

# reset the index to prevent errors in knn method during indexing elements
X_train = X_train.reset_index(drop=True) 
y_train = y_train.reset_index(drop=True)

# standarization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# class to evaluate all used models
class ModelsEvaluation():
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    # plot accuracy score based on maximum depth
    def tree_evaluation(self, max_depth_to_analyse):
        tree_score = []
        depth_list = [i for i in range(1, max_depth_to_analyse + 1)]
        for depth in depth_list:
            tree_score.append(decision_tree_classifier(depth, self.X_train, self.y_train, self.X_test, self.y_test)[0])
        plt.plot(depth_list, tree_score)
        plt.xlabel('Maximum depth')
        plt.ylabel('Prediction accuracy')
        plt.title('Decision tree classifier')
        plt.show()


    # plot accuracy score based on maximum number of iterations
    def logistic_evaluation(self, max_iter_to_analyse):
        log_score = []
        iter_list = [i for i in range(1, max_iter_to_analyse + 1)]
        for max_iter in iter_list:
            log_score.append(logistic_regression(max_iter, self.X_train, self.y_train, self.X_test, self.y_test)[0])
        plt.plot(iter_list, log_score)
        plt.xlabel('Max iteration')
        plt.ylabel('Prediction accuracy')
        plt.title('Logistic regression')
        plt.show()


    # plot accuracy score based on number of k nearest neighbours
    def knn_evaluation(self, max_k_to_analyse):
        knn_score = []
        k_list = [i for i in range(1, max_k_to_analyse + 1)]
        for k in k_list:
            knn_score.append(knn(k, self.X_train, self.y_train, self.X_test, self.y_test)[0])
        plt.plot(k_list, knn_score)
        plt.xlabel('K')
        plt.ylabel('Prediction accuracy')
        plt.title('K nearest neighbour')
        plt.show()


    # get the best set of hyperparameters and the best model, user can input own hyperparameters, but there are default one as well
    def nn_best_hyperparameters(self, param_distributions={
                                        'hidden_layer1_neurons': [32, 64, 128],
                                        'hidden_layer2_neurons': [32, 64, 128],
                                        'dropout': [0.2, 0.3, 0.4],
                                        'learning_rate': [0.001, 0.01, 0.1],
                                                            }):
        model = KerasClassifier(build_fn=create_nn_model_eval, verbose=0)

        # choose hyperparameters randomly
        search = RandomizedSearchCV(model, param_distributions, n_iter=10, cv=3)
        search.fit(X_train, y_train, epochs=10, validation_split=0.1)
        print(search.best_params_)
        best_model = search.best_estimator_.model

        # plot the training curves for the best model
        history = best_model.fit(self.X_train, self.y_train, epochs=20, validation_split=0.1)
        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(accuracy))
        
        plt.plot(epochs, accuracy, 'r', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        
        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()

models_evaluation = ModelsEvaluation(X_train, y_train, X_test, y_test)

