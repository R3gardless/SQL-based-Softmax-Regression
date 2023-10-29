# MNIST Softmax Regression With SQL

import os

# single threaded or use mkl
single = False
if single:
    threads_amount = "1"
else:
    threads_amount = "36"
os.environ["MKL_NUM_THREADS"] = threads_amount
os.environ["NUMEXPR_NUM_THREADS"] = threads_amount
os.environ["OMP_NUM_THREADS"] = threads_amount

from timeit import default_timer as timer
from datetime import datetime
from tableauhyperapi import HyperProcess, Telemetry, Connection, CreateMode, escape_string_literal

import numpy as np
import matplotlib.pyplot as plt
import psycopg2 as psy
import pandas as pd


def create_dataset(train_filename, test_filename):
    """
    Create MNIST dataset 

    <return> (x_train, y_train, x_test, y_test)
    """
    
    train_data = pd.read_csv(train_filename, header=None).to_numpy()
    test_data = pd.read_csv(test_filename, header=None).to_numpy()
    
    x_train = train_data[:, 1:]
    x_train = x_train / 255.0
    y_train = train_data[:, 0]
    
    x_test = test_data[:, 1:]
    x_test = x_test / 255.0
    y_test = test_data[:, 0]

    return x_train, y_train, x_test, y_test

def sample_dataset(X, y, samples):
    """
    MNIST dataset sampling

    <return> (x, y)
    """
    
    X = X[:samples]
    y = y[:samples]

    return X, y

def get_queries_CSV_COO(X, y, hyper=True):
    """
    Creates the database queries for inserting the data in the COO format via CSV copy
  
    <return> list of string queries
    """
    COO_X = list()
    COO_y = list()
    COO_weight = list()
    for row, value in enumerate(y):
        COO_y.append([row, value])
    for row, x in enumerate(X):
        for col, value in enumerate(x):
            if value: COO_X.append([row, col, value])

    for row in range(784):
        for col in range(10):
            COO_weight.append([row, col])

    X_file = "./dataset/X.csv"
    y_file = "./dataset/Y.csv"
    weight_file = "./dataset/weight.csv"

    np.savetxt(X_file, np.array(COO_X), '%i %i %f')
    np.savetxt(weight_file, np.array(COO_weight), '%i %i')
    np.savetxt(y_file, np.array(COO_y), '%i %f')


    # create the queries for insert
    queries = [
        "DROP TABLE IF EXISTS M;",
        "CREATE TEMPORARY TABLE M (r INTEGER , c INTEGER );",
        "DROP TABLE IF EXISTS Y;",
        "CREATE TEMPORARY TABLE Y (r INTEGER , val DOUBLE PRECISION );",
        "DROP TABLE IF EXISTS X;",
        "CREATE TEMPORARY TABLE X (r INTEGER , c INTEGER , val DOUBLE PRECISION );",
    ]

    if hyper:
        csv_insert = [
            f"COPY Y FROM {escape_string_literal(y_file)} WITH (format csv, delimiter ' ');",
            f"COPY weight FROM {escape_string_literal(weight_file)} WITH (format csv, delimiter ' ');",
            f"COPY X FROM {escape_string_literal(X_file)} WITH (format csv, delimiter ' ');"
        ]
    else:
        csv_insert = [
            f"COPY Y FROM {escape_string_literal(os.path.abspath(y_file))} WITH DELIMITER ' ';",
            f"COPY weight FROM {escape_string_literal(os.path.abspath(weight_file))} WITH DELIMITER ' ';",
            f"COPY X FROM {escape_string_literal(os.path.abspath(X_file))} WITH DELIMITER ' ';"
        ]
    # add the specific queries to the list of all queries
    for query in csv_insert:
        queries.append(query)

    return queries

def get_gradient_descent_query(parameter=None):
    
    iterations = parameter.get('iterations')
    regularization = parameter.get('regularization')
    step_width = parameter.get('step_width')
    num_outputs = parameter.get('num_outputs')

    with open(os.path.join('queries', f'softmax_regression_train.sql')) as f:
        query = f.read().format(
            num_outputs=num_outputs,
            iterations=iterations,
            bias=regularization,
            step_width=step_width,
        )

    return query

############
# POSTGRES #
############
def postgres_experiment(X, y, parameter=None):

    # get connection to the database
    conn = psy.connect(user='postgres', password='pw', database='postgres', host='localhost')
    conn.set_isolation_level(psy.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    epochs = parameter.get('epochs')
    iterations = parameter.get('iterations')
    verbose = parameter.get('verbose')
    num_outputs = parameter.get('num_outputs')
    features = parameter.get('features')

    # create tables and insert data
    queries = get_queries_CSV_COO(X, y, hyper=False)

    tic = timer()
    for query in queries:
        try:
            cur.execute(query)
        except Exception as e:
            print(e)
            return np.nan, None, None
    toc = timer()
    
    if verbose: print(f"[VERBOSE] Postgres Insert data in {toc - tic}s.")
    conn.commit()
    # execute query
    query = get_gradient_descent_query(parameter=parameter)

    # burn in
    for _ in range(epochs):
        try:
            cur.execute(query)
            toc = timer()
        except Exception:
            return np.nan, None, None
    tic = timer()
    for _ in range(epochs):
        try:
            cur.execute(query)
            toc = timer()
        except Exception:
            return np.nan, None, None
    toc = timer()
    result = cur.fetchall()
    conn.close()

    COO_weight_len = len(list(result))
    w = [[0.0 for j in range(num_outputs)] for i in range(features)]
    for i in range(len(result)):
        w[result[i][0]][result[i][1]] = result[i][2]
    a = float(result[0][3])
    runtime = (iterations * epochs) / (toc - tic)

    return runtime, w, COO_weight_len, a
    

#########
# HYPER #
#########
def hyper_experiment(X, y, parameter=None):
    parameters = {
        "log_config": "",
        "initial_compilation_mode": "o",  # o, v, c
        "max_query_size": "10000000000",
        "hard_concurrent_query_thread_limit": threads_amount
    }
    
    epochs = parameter.get('epochs')
    iterations = parameter.get('iterations')
    verbose = parameter.get('verbose')
    num_outputs = parameter.get('num_outputs')
    features = parameter.get('features')

    with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU, parameters=parameters) as hyper:
        with Connection(hyper.endpoint, f'data.hyper', CreateMode.CREATE_AND_REPLACE) as connection:
            # create tables and insert data
            queries = get_queries_CSV_COO(X, y, hyper=True)
            tic = timer()
            for query in queries:
                connection.execute_list_query(query)
            toc = timer()
            if verbose: print(f"[VERBOSE] HyPer Insert data in {toc - tic}s.")

            # execute query
            query = get_gradient_descent_query(parameter=parameter)
            # burn in
            for _ in range(epochs):
                result = connection.execute_list_query(query)
            tic = timer()
            for _ in range(epochs):
                result = connection.execute_list_query(query)
            toc = timer()

    if result: 
        COO_weight_len = len(list(result))
        w = [[0.0 for j in range(num_outputs)] for i in range(features)]
        for i in range(len(result)):
            w[result[i][0]][result[i][1]] = result[i][2]
        a = float(result[0][3])
        runtime = (iterations * epochs) / (toc - tic)

        return runtime, w, COO_weight_len, a

    return None, None, None


#########
# NUMPY #
#########
def numpy(X, y, parameter):
    epochs = parameter.get('epochs')
    iterations = parameter.get('iterations')
    regularization = parameter.get('regularization')
    step_width = parameter.get('step_width')
    num_outputs = parameter.get('num_outputs')
    
    def softmax(X, w, bias):
        class_scores = X.dot(w) + bias
        exp_scores = np.exp(class_scores)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # size example : (60000, 10)

    def gradient(X, w, bias, y):
        probs = softmax(X, w, bias)
        diff = probs - np.eye(probs.shape[1])[y] # size example : (60000, 10)
        w_grad = (1.0 / num_outputs) * X.T.dot(diff) # size example : (784, 60000) x (60000, 10)
        return w_grad

    def gradient_descent(X, y, bias, iterations):
        w = np.zeros((X.shape[1], num_outputs))
        alpha = step_width
        for i in range(iterations):
            g = gradient(X, w, bias, y)
            w = w - alpha * g
        return w
    
    # predict classes of samples in X, use weights w
    def predict(X, w, bias):
        probs = softmax(X, w, bias)
        return np.argmax(probs, axis=1)

    # burnin
    for _ in range(epochs):
        w = gradient_descent(X, y, bias=regularization, iterations=iterations)
        predicts = predict(X, w, bias=regularization)
        a = np.mean(predicts == y)
    tic = timer()
    for _ in range(epochs):
        w = gradient_descent(X, y, bias=regularization, iterations=iterations)
        predicts = predict(X, w, bias=regularization)
        a = np.mean(predicts == y)
    toc = timer()

    return (iterations * epochs) / (toc-tic), w, a


if __name__ == "__main__":

    ### CREATE DATA ###
    X_train, Y_train, x_test, y_test = create_dataset('./dataset/mnist_train.csv', './dataset/mnist_test.csv')

    ### PARAMETER FOR THE OPTIMIZATION ###
    parameter = {
        'epochs': 100,
        'iterations': 100,
        'regularization': 2,
        'step_width': 0.001,
        'features': X_train.shape[1],
        'num_outputs' : len(np.unique(Y_train)),
        'verbose': True,
        'samples': 0,
    }

    for samples in [60, 600, 6000, 60000]:
        parameter['samples'] = samples        
        x_train, y_train = sample_dataset(X_train, Y_train, samples)
        print(f"{datetime.today().strftime('%Y-%m-%d %H:%M:%S')} With {samples} Experiment Start!\n")


        # Numpy Train Part
        print(f"{datetime.today().strftime('%Y-%m-%d %H:%M:%S')} Train with Numpy!\n")
        runtime, wn, an = numpy(x_train, y_train, parameter)
        print(f"numpy info : {runtime} {len(wn)} {an}\n")

        # HyPer Train Part
        print(f"{datetime.today().strftime('%Y-%m-%d %H:%M:%S')} Train with HyPer!\n")
        runtime, wh, COO_weight_len, ah = hyper_experiment(x_train, y_train, parameter=parameter)
        print(f"hyper info : {runtime} {len(wh)} {ah}\n")
        
        # PostgreSQL Train Part
        print(f"{datetime.today().strftime('%Y-%m-%d %H:%M:%S')} Train with Postgres!\n")
        runtime, wp, COO_weight_len, ap = postgres_experiment(x_train, y_train, parameter=parameter)
        print(f"postgres info : {runtime}, {COO_weight_len}, {len(wp)}, {ap}\n")
