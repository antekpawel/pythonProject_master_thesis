# This is a sample Python script.
import pandas as pd
import numpy as np

np.random.seed(42)

x = np.array([1.4, 0.7])
y_true = np.array([58])

def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.rand(n_h, n_x)
    W2 = np.random.rand(n_h, n_y)
    return W1, W2
def forward_propagation(X, W1, W2):
    H1 = np.dot(X, W1)
    y_pred = np.dot(H1, W2)
    return H1, y_pred

def cal_err(y_pred, y_true):
    return y_pred - y_true

def predict(X, W1, W2):
    _, y_pred = forward_propagation(X, W1, W2)
    return y_pred[0]

def backpropagation (x, w1, w2, learning_rate, iters = 1000, precision = 1e-6):
    H1, y_pred = forward_propagation(x, w1, w2)
    train_loss  =[]

    for i in range(iters):
        error = cal_err(y_pred, y_true)
        w2 = w2 - learning_rate * error * H1.T
        w1 = w1 - learning_rate * error * np.dot(x.T, w2.T)

        y_pred = predict(x, w1, w2)
        print(f'Iter #{i}: y_pred {y_pred}: loss: {abs(cal_err(y_pred, y_true))}')
        train_loss.append(abs(cal_err(y_pred, y_true[0])))

        if abs(error) < precision:
            break

    return w1, w2, train_loss

def build_model():
    w1, w2 = initialize_parameters(2, 2, 1)
    w1, w2, train_loss = backpropagation(x, w1, w2, 0.001)
    model = {'W1': w1, 'W2': w2, 'train_loss': train_loss}

    return model


model = build_model()


loss = pd.DataFrame({'train_loss': model['train_loss']})
loss = loss.reset_index().rename(columns={'index': 'iter'})
loss['iter'] += 1
loss.head()

import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter(x=loss['iter'], y=loss['train_loss'], mode='markers+lines'))
fig.show()









