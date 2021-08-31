import numpy as np
import pandas as pd
import math
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

x_range = np.arange(-5, 5, 0.1)

y_lin = x_range
fig_lin = px.line(x=x_range, y=y_lin, title='Linear')
fig_lin.show()
fig_lin.write_image("images/AF_lin.svg")

y_tanh = np.tanh(x_range)
fig_tanh = px.line(x=x_range, y=y_tanh, title='tanh')
fig_tanh.show()
fig_tanh.write_image("images/AF_tanh.svg")

y_exp = np.exp(x_range)
fig_exp = px.line(x=x_range, y=y_exp, title='Exponential')
fig_exp.show()
fig_exp.write_image("images/AF_exp.svg")

y_relu = x_range.copy()
y_relu[y_relu < 0] = 0
fig_relu = px.line(x=x_range, y=y_relu, title='ReLU')
fig_relu.show()
fig_relu.write_image("images/AF_relu.svg")

y_sig = 1 / (1 + np.exp(x_range * (-1)))
fig_sig = px.line(x=x_range, y=y_sig, title='Sigmoid')
fig_sig.show()
fig_sig.write_image("images/AF_sig.svg")

# avarage = [82.6, 82.0, 87.7, 87.8, 88.3, 116.2, 159.9, 218.6]
# param = [29, 25, 21, 17, 13, 9, 5, 3]
# x_data = ['2-7-1', '2-6-1', '2-5-1', '2-4-1', '2-3-1', '2-2-1', '2-1-1', '2-1']
# df = pd.DataFrame({'Epochs':[82.6, 82.0, 87.7, 87.8, 88.3, 116.2, 159.9, 218.6],
#                    'No.parameters':[29, 25, 21, 17, 13, 9, 5, 3]}, index=x_data)
# print(df)
#
# # Create figure with secondary c_C-axis
# fig = make_subplots(specs=[[{"secondary_y": True}]])
#
# # Add traces
# fig.add_trace(
#     go.Scatter(x=x_data, y=avarage, name="Epoki"),
#     secondary_y=False,
# )
#
# fig.add_trace(
#     go.Scatter(x=x_data, y=param, name="Ilosc parametrow"),
#     secondary_y=True,
# )
#
# # Set x-axis title
# fig.update_xaxes(title_text="Struktura sieci")
#
# # Set c_C-axes titles
# fig.update_yaxes(title_text="Epoki", secondary_y=False)
# fig.update_yaxes(title_text="Ilosc parametrow", secondary_y=True)
#
# fig.show()
# fig.write_image("images/NoNeurons.svg")
#
# with pd.ExcelWriter('Properties.xlsx') as writer:
#     df.to_excel(writer, sheet_name='Testy')
