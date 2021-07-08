import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    return math.sqrt(2 * ca0 * cb0 * eq ** 2
                     - ca0 ** 2 * eq ** 2
                     - cb0 ** 2 * eq ** 2
                     - 4 * ca0 * cb0 * eq)


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

Temperature = data_IP['T_r [C]'][0]
c_A0 = data_IP['c_A0 [mol/m3]'][0]
c_B0 = data_IP['c_B0 [mol/m3]'][0]

k2 = reaction_rate(Temperature)
Ke = equilibrium(Temperature)

# sqrt_val = pierwiastek(c_A0, c_B0, Ke)
eq_concetration = stezenie_row(c_A0, c_B0, Temperature)
print(eq_concetration)
# numerator = - c_A0 * Ke - c_B0 * Ke
# finish_time = 2 * Ke / sqrt_val * math.atan((numerator + 2 * (Ke - 1) * eq_concetration) / sqrt_val) / k2
# print(finish_time)

reaction_time = np.linspace(0, 4 * 3600)
c_C = odeint(r_reaction_cd, 0, reaction_time, args=(c_A0, c_B0, Ke, k2))

# Table of output data
con_in_time = pd.DataFrame(list(zip(reaction_time, np.hstack(c_C))), columns=['Czas [s]', 'c_C = c_D [mol/m3]'])
con_in_time['c_A [mol/m3]'] = c_A0 - con_in_time['c_C = c_D [mol/m3]']
con_in_time['c_B [mol/m3]'] = c_B0 - con_in_time['c_C = c_D [mol/m3]']
con_in_time['r [mol/m3/s]'] = k2 * (con_in_time['c_A [mol/m3]'] * con_in_time['c_B [mol/m3]']
                                    - con_in_time['c_C = c_D [mol/m3]'] ** 2 / Ke)

print(con_in_time)

# Plot data
fig_concetration = px.line(con_in_time,
                           x='Czas [s]',
                           y=['c_B [mol/m3]', 'c_A [mol/m3]', 'c_C = c_D [mol/m3]'],
                           labels=dict(x="Czas [s]", y="Stezenie [mol/m3]", color="Skladniki"))
fig_concetration.show()
fig_concetration.write_image('images/Concetration Tr ' + str(data_IP['T_r [C]'][0])
                             + ' c_A0 ' + str(data_IP['c_A0 [mol/m3]'][0])
                             + ' c_B0 ' + str(data_IP['c_B0 [mol/m3]'][0])
                             + '.svg')

# Check the value
print(c_C[-1] ** 2 / (40 - c_C[-1]) / (20 - c_C[-1]) / equilibrium(20))

# Save data
dataset = pd.concat([dataSet, fluidProp], ignore_index=True)
with pd.ExcelWriter('Concetration_data.xlsx') as writer:
    dataset.to_excel(writer,
                     sheet_name='Example Tr ' + str(data_IP['T_r [C]'][0])
                                + ' c_A0 ' + str(data_IP['c_A0 [mol/m3]'][0])
                                + ' c_B0 ' + str(data_IP['c_B0 [mol/m3]'][0]))
