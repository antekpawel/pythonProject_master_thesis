import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    model = Sequential()
    model.add(Dense(4, input_dim=2))
    model.add(Dense(1, activation='relu'))

    model.compile(optimizer='SGD',
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


def get_ideal_gas_data():
    np.random.seed(1)
    pressure = np.random.rand(100) * 9.9 + 0.1
    temperature = np.random.rand(100) * 200 + 273.15
    density = pressure * 1e6 / temperature / 8.314 * 1 / 1000

    w_p = (pressure - 0.1) / 9.9
    w_t = (temperature - 273.15) / 200
    d_min = 0.1 * 1e6 / 473.15 / 8.314 * 1 / 1000
    d_max = 10 * 1e6 / 273.15 / 8.314 * 1 / 1000
    w_d = (density - d_min) / (d_max - d_min)

    for i in w_d:
        if i > 1 or i < 0:
            print('err')

    ideal_gas_data_f = pd.DataFrame(list(zip(pressure, temperature, density, w_p, w_t, w_d)),
                                    columns=['Pressure [MPa]', 'Temperature [K]',
                                             'Density [kg/m3]', 'W_p [-]', 'W_t [-]', 'W_d [-]'])

    ideal_gas_data_f.to_excel('Ideal_gas_data.xlsx')
    return ideal_gas_data_f


def save_figure(vector, name):
    fig_err_abs = go.Figure()
    fig_err_abs.add_trace(go.Scatter(y=vector, mode='markers'))
    fig_err_abs.update_layout(xaxis_title='No.', yaxis_title=name)
    fig_err_abs.show()
    fig_err_abs.write_image("images/{0}.svg".format(re.sub('\W+', '', name)))


try:
    with open('Ideal_gas_data.xlsx', encoding='utf-8', errors='ignore') as f:
        ideal_gas_data = pd.read_excel("Ideal_gas_data.xlsx", index_col=0)
except IOError:
    ideal_gas_data = get_ideal_gas_data()

dataset = ideal_gas_data.copy()

fig = px.scatter_matrix(dataset, dimensions=['Pressure [MPa]', 'Temperature [K]', 'Density [kg/m3]'], height=700)
fig.show()
fig.write_image("images/Relation_chart.svg")

dataset = dataset[['Pressure [MPa]', 'Temperature [K]', 'Density [kg/m3]']]
train_dataset = dataset.sample(frac=0.7, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop('Density [kg/m3]')
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('Density [kg/m3]')
test_labels = test_dataset.pop('Density [kg/m3]')

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

normed_test_data = normed_test_data.values
normed_train_data = normed_train_data.values

filepath = 'Best_weights.hdf5'
checkpoint = ModelCheckpoint(filepath=filepath, monitor='mse', verbose=0, save_best_only=True, mode='min')
es = EarlyStopping(monitor='mse', mode='min', verbose=1, patience=8)

model = build_model()
history = model.fit(normed_train_data, train_labels.values,
                    epochs=500,
                    validation_split=0.2,
                    verbose=0,
                    batch_size=32,
                    callbacks=[checkpoint, es])
plot_hist(history)
print(model.summary())

test_predictions = model.predict(normed_test_data).flatten()
data_compare = pd.DataFrame(test_labels)
data_compare['Predicted density [kg/m3]'] = test_predictions
data_compare['Error [kg/m3]'] = abs(test_labels - test_predictions)
data_compare['Error [%]'] = data_compare['Error [kg/m3]'] / test_labels
data_compare = data_compare.reset_index(drop=True)
data_compare = data_compare.sort_values(by="Density [kg/m3]", ignore_index=True)

save_figure(data_compare['Error [%]'], 'Error [%]')
save_figure(data_compare['Error [kg/m3]'], 'Error [kg/m^3]')

rmse = get_rmse(history)
r2 = pd.DataFrame({'r2': [r2_score(test_labels, test_predictions)]})

with pd.ExcelWriter('Properties.xlsx') as writer:
    data_compare.to_excel(writer, sheet_name='Data_compare')
    dataset.head().to_excel(writer, sheet_name='Head')
    pd.DataFrame(model.get_weights()).to_excel(writer, sheet_name='Weights')
    pd.DataFrame(rmse).to_excel(writer, sheet_name='RMSE')
    r2.to_excel(writer, sheet_name='r2')
print(r2)
