# open_x_task

<b>'Data'</b> folder contains data for classification problem. <br>
<b>'ml_models.py'</b> contains required ML models: heuristic (kNN), 2 baseline models (Decision Tree, Logistic Regression) and a neural network model.<br>
<b>'rest_api.py'</b> contains REST API in Flask, user chooses one of four models, inputs parameters and gets accuracy score, confusion matrix and downloadable csv file with predicted labels.<br>
<b>'templates'</b> folder contains all necessary html templates for REST API<br>
<b>'models_evaluations.py'</b> contains a one class that defines a method for each model to plot accuracy metric based on given range of parameters for three models, and one method for neural network to evaluate and pick best hyperparameters<br>
