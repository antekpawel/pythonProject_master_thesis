import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import re
from sklearn.metrics import r2_score

if not os.path.exists("images"):
    os.mkdir("images")


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


def build_model():
    tensorflow.random.set_seed(2)
    model = Sequential()
    # model.add(Dense(5, activation='sigmoid', input_dim=4))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mse'])
    return model


def plot_hist(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist['rmse'] = np.sqrt(hist['mse'])
    hist['val_rmse'] = np.sqrt(hist['val_mse'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['rmse'], name='Zbior uczacy', mode='markers+lines'))
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_rmse'], name='Zbior walidacyjny', mode='markers+lines'))
    fig.update_layout(width=1000, height=500, xaxis_title='Epoki',
                      yaxis_title='Blad sredniokwadratowy', yaxis_type='log')
    fig.show()
    fig.write_image("images/RMSE_chart.svg")


def get_rmse(history):
    hist = pd.DataFrame(history.history)
    hist['rmse'] = np.sqrt(hist['mse'])
    return hist['rmse']


def save_figure(vector, name):
    fig_err_abs = go.Figure()
    fig_err_abs.add_trace(go.Scatter(y=vector, mode='markers'))
    fig_err_abs.update_layout(xaxis_title='No.', yaxis_title=name)
    fig_err_abs.show()
    fig_err_abs.write_image("images/{0}.svg".format(re.sub('\W+', '', name)))


try:
    with open('Concetration_data.xlsx', encoding='utf-8', errors='ignore') as f:
        ideal_gas_data = pd.read_excel("Concetration_data.xlsx", index_col=0)
except IOError:
    print('Error, no such file!')

dataset = ideal_gas_data.copy()

fig = px.scatter_matrix(dataset,
                        dimensions=['c_C [mol/m3]', 'c_A [mol/m3]', 'c_B [mol/m3]',
                                    'Reaction temperature [C]', 'r [mol/m3/s]'],
                        height=900, width=900)
fig.show()
fig.write_image("images/Relation_chart.svg")

# dataset = dataset[['Pressure [MPa]', 'Temperature [K]', 'Density [kg/m3]']]
train_dataset = dataset.sample(frac=0.7, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop('r [mol/m3/s]')
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('r [mol/m3/s]')
test_labels = test_dataset.pop('r [mol/m3/s]')

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

normed_test_data = normed_test_data.values
normed_train_data = normed_train_data.values

filepath = 'Best_weights.hdf5'
checkpoint = ModelCheckpoint(filepath=filepath, monitor='mse', verbose=0, save_best_only=True, mode='min')
es = EarlyStopping(monitor='mse', mode='min', verbose=1, patience=25)

model = build_model()
history = model.fit(normed_train_data, train_labels.values,
                    epochs=20000,
                    validation_split=0.2,
                    verbose=0,
                    batch_size=32,
                    callbacks=[es, checkpoint])
# plot_hist(history)
print(model.summary())

test_predictions = model.predict(normed_test_data).flatten()
data_compare = pd.DataFrame(test_labels)
data_compare['Predicted r [mol/m3/s]'] = test_predictions
data_compare['Error [mol/m3/s]'] = abs(test_labels - test_predictions)
data_compare['Error [%]'] = data_compare['Error [mol/m3/s]'] / test_labels * 100
data_compare = data_compare.reset_index(drop=True)
data_compare = data_compare.sort_values(by="r [mol/m3/s]", ignore_index=True)

# save_figure(data_compare['Error [%]'], 'Blad [%]')
# save_figure(data_compare['Error [mol/m3/s]'], 'Blad [mol/m3/s]')

rmse = get_rmse(history)
r2 = pd.DataFrame({'r2': [r2_score(test_labels, test_predictions)]})

with pd.ExcelWriter('Properties.xlsx') as writer:
    data_compare.to_excel(writer, sheet_name='Data_compare')
    dataset.head().to_excel(writer, sheet_name='Head')
    pd.DataFrame(model.get_weights()).to_excel(writer, sheet_name='Weights')
    pd.DataFrame(rmse).to_excel(writer, sheet_name='RMSE')
    r2.to_excel(writer, sheet_name='r2')
print(r2)
print(sum(data_compare['Error [%]'] < 1) / len(data_compare))
print(sum(data_compare['Error [%]'] < 10) / len(data_compare))
print(sum(data_compare['Error [%]'] < 50) / len(data_compare))
print(sum(data_compare['Error [%]'] < 100) / len(data_compare))
