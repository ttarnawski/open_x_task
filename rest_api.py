from flask import Flask, request, render_template, send_file, session, url_for
import csv, io
from io import StringIO
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ml_models import decision_tree_classifier, logistic_regression, knn, create_nn_model, buffer_plot


app = Flask(__name__)
app.secret_key = 'abc'

### load data ###
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
##################


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        # get the value of the selected radio button
        choice = request.form['choice']
        
        # render the appropriate template based on the user's choice
        if choice == 'knn':
            return render_template('knn.html')
        elif choice == 'tree':
            return render_template('tree.html')
        elif choice == 'logistic':
            return render_template('logistic.html')
        elif choice == 'nn':
            return render_template('nn.html')
    
    return render_template('index.html')


### functions to run specific algorithm
@app.route('/run_tree', methods=['POST'])
def run_tree():
    # get the user provided parameters from the request form
    if request.method == 'POST':
        max_depth = int(request.form['max_depth'])
        result, disp, y_pred = decision_tree_classifier(max_depth, X_train, y_train, X_test, y_test)
        confusion_matrix = buffer_plot(disp)
        session['y_pred'] = y_pred.tolist()
    
    # after filling the form render the template that shows accuracy score of an algorithm joined by confusion matrix
    return render_template('result.html', result=result, image_data=confusion_matrix)


@app.route('/run_logistic', methods=['POST'])
def run_logistic():
    if request.method == 'POST':
        max_iter = int(request.form['max_iter'])
        result, disp, y_pred = logistic_regression(max_iter, X_train, y_train, X_test, y_test)
        confusion_matrix = buffer_plot(disp)
        session['y_pred'] = y_pred.tolist()

    return render_template('result.html', result=result, image_data=confusion_matrix)


@app.route('/run_knn', methods=['POST'])
def run_knn():
    if request.method == 'POST':
        k = int(request.form['k'])
        result, disp, y_pred = knn(k, X_train, y_train, X_test, y_test)
        confusion_matrix = buffer_plot(disp)
        session['y_pred'] = y_pred.tolist()

    return render_template('result.html', result=result, image_data=confusion_matrix)


@app.route('/run_nn', methods=['POST'])
def run_nn():
    if request.method == 'POST':
        layer1 = int(request.form['layer1'])
        layer2 = int(request.form['layer2'])
        dropout = float(request.form['dropout'])
        learning_rate = float(request.form['learning_rate'])
        epochs = int(request.form['epochs'])
        result, disp, y_pred = create_nn_model(layer1, layer2, dropout, learning_rate, epochs, X_train, y_train, X_test, y_test)
        confusion_matrix = buffer_plot(disp)
        session['y_pred'] = y_pred.tolist()

    return render_template('result.html', result=result, image_data=confusion_matrix)


@app.route('/download_prediction', methods=['GET'])
def download_prediction():
    # get the predicted values
    y_pred = session.get('y_pred', None)

    # save y_pred as a CSV file
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Predicted labels'])
    for label in y_pred:
        writer.writerow([label])
    
    # send the CSV file that user can download
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='labels_prediction.csv')