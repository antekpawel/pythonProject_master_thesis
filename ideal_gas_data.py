import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping


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
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['rmse'], name='rmse', mode='markers+lines'))
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_rmse'], name='val_rmse', mode='markers+lines'))
    fig.update_layout(width=1000, height=500, title='RMSE vs. VAL_RMSE', xaxis_title='Epoki',
                      yaxis_title='Root Mean Squared Error', yaxis_type='log')
    fig.show()


def get_rmse(history):
    hist = pd.DataFrame(history.history)
    hist['rmse'] = np.sqrt(hist['mse'])
    print(hist['rmse'])


def get_ideal_gas_data():
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
                                  columns=['Pressure [MPa]', 'Temperature [K]', 'Density [kg/m3]',
                                           'W_p [-]', 'W_t [-]', 'W_d [-]'])

    ideal_gas_data_f.to_excel('Ideal_gas_data.xlsx')
    return ideal_gas_data_f


try:
    with open('Ideal_gas_data.xlsx', encoding='utf-8', errors='ignore') as f:
        ideal_gas_data = pd.read_excel("Ideal_gas_data.xlsx", index_col=0)
except IOError:
    ideal_gas_data = get_ideal_gas_data()

dataset = ideal_gas_data.copy()

fig = px.scatter_matrix(dataset, dimensions=['Pressure [MPa]', 'Temperature [K]'], height=700)
fig.show()

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
es = EarlyStopping(monitor='mse', mode='min', verbose=1, patience=3)

model = build_model()
history = model.fit(normed_train_data, train_labels.values,
                    epochs=500,
                    validation_split=0.2,
                    verbose=0,
                    batch_size=32,
                    callbacks=[checkpoint, es])
plot_hist(history)

test_predictions = model.predict(normed_test_data).flatten()
data_compare = pd.DataFrame(test_labels)
data_compare['Predicted density [kg/m3]'] = test_predictions
data_compare = data_compare.reset_index(drop=True)

print(data_compare)
data_compare.to_excel('Data_compare.xlsx')

table_ptro = dataset.head()
print(table_ptro)
table_ptro.to_excel('Head.xlsx')

print(model.layers[0].get_weights()[0])
print(model.layers[0].get_weights()[1])
