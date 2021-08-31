import math
import numpy as np
import pandas as pd
import plotly.express as px
import os
from scipy.integrate import odeint

if not os.path.exists("images"):
    os.mkdir("images")


def reaction_rate(reaction_tem):
    return 0.03 * math.e ** (-2e4 / 8.314 / (reaction_tem + 273.15))


def equilibrium(reaction_tem):
    return 2.3 * math.e ** (-9e2 / 8.314 / (reaction_tem + 273.15))


def r_reaction_cd(y, t, ca0, cb0, eq, rr):
    return rr * ((ca0 - y) * (cb0 - y) - y ** 2 / eq)


def pierwiastek(ca0, cb0, eq):
    return math.sqrt(+ ca0 ** 2 * eq
                     + cb0 ** 2 * eq
                     - 2 * ca0 * cb0 * (eq - 2))


def stezenie_row(ca0, cb0, tem):
    a = 1 - 1 / equilibrium(tem)
    b = - ca0 - cb0
    c = ca0 * cb0
    d = math.sqrt(b ** 2 - 4 * a * c)
    return (- b - d) / 2 / a


try:
    with open('Data_IP.xlsx', encoding='utf-8', errors='ignore') as f:
        data_IP = pd.read_excel("Data_IP.xlsx", index_col=0)
except IOError:
    print("No file!")

dataset = pd.DataFrame()
stop_condition = data_IP.copy()
stop_condition['Stezenie row [mol/m3]'] = 0
stop_condition['Stezenie row [mol/m3]'] = stop_condition['Stezenie row [mol/m3]'].astype(float)
stop_condition['Time to stop [s]'] = 0
stop_condition['Equilibrium'] = 0
stop_condition['Equilibrium'] = stop_condition['Equilibrium'].astype(float)
plot = False

for index, row in data_IP.iterrows():
    i = index
    # Set input data
    Temperature = data_IP['T_r [C]'][i]
    c_A0 = data_IP['c_A0 [mol/m3]'][i]
    c_B0 = data_IP['c_B0 [mol/m3]'][i]
    k2 = reaction_rate(Temperature)
    Ke = equilibrium(Temperature)

    # Information
    sqrt_val = pierwiastek(c_A0, c_B0, Ke)
    eq_concetration = stezenie_row(c_A0, c_B0, Temperature)
    print(eq_concetration)
    numerator = c_A0 * Ke + c_B0 * Ke
    arctan = 1 - ((numerator - 2 * (Ke - 1) * eq_concetration * 0.95) / sqrt_val / math.sqrt(Ke) - 1)
    arctan = 0.99
    print(arctan)
    finish_time = 2 * math.sqrt(Ke) / sqrt_val / k2 * math.log((1 + arctan) / (1 - arctan)) / 2
    print(finish_time)

    # Numerical output
    reaction_time = np.linspace(0, finish_time)
    c_C = odeint(r_reaction_cd, 0, reaction_time, args=(c_A0, c_B0, Ke, k2))

    # Table of output data
    con_in_time = pd.DataFrame(list(zip(np.hstack(c_C))), columns=['c_C [mol/m3]'])
    con_in_time['c_D [mol/m3]'] = con_in_time['c_C [mol/m3]']
    con_in_time['c_A [mol/m3]'] = c_A0 - con_in_time['c_C [mol/m3]']
    con_in_time['c_B [mol/m3]'] = c_B0 - con_in_time['c_C [mol/m3]']
    con_in_time['Reaction temperature [C]'] = Temperature
    con_in_time['r [mol/m3/s]'] = k2 * (con_in_time['c_A [mol/m3]'] * con_in_time['c_B [mol/m3]']
                                        - con_in_time['c_C [mol/m3]'] ** 2 / Ke)

    # Plot data
    if plot:
        fig_concetration = px.line(con_in_time,
                                   x='Czas [s]',
                                   y=['c_B [mol/m3]', 'c_A [mol/m3]', 'c_C [mol/m3]'],
                                   labels=dict(x="Czas [s]", y="Stezenie [mol/m3]", color="Skladniki"))
        fig_concetration.show()
        fig_concetration.write_image('images/Concetration Tr ' + str(data_IP['T_r [C]'][i])
                                     + ' c_A0 ' + str(data_IP['c_A0 [mol/m3]'][i])
                                     + ' c_B0 ' + str(data_IP['c_B0 [mol/m3]'][i])
                                     + '.svg')

    # Check the value
    check_eq = c_C[-1] ** 2 / (c_A0 - c_C[-1]) / (c_B0 - c_C[-1]) / equilibrium(Temperature)

    # Save data
    dataset = pd.concat([dataset, con_in_time], ignore_index=True)
    stop_condition['Stezenie row [mol/m3]'][i] = eq_concetration.astype(float)
    stop_condition['Time to stop [s]'][i] = finish_time
    stop_condition['Equilibrium'][i] = check_eq

print(dataset)
print(stop_condition)
with pd.ExcelWriter('Concetration_data.xlsx') as writer:
    dataset.to_excel(writer, sheet_name='Concetration_data')
    stop_condition.to_excel(writer, sheet_name='Stop_condition')
