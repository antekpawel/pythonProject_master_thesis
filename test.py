import numpy as np
import plotly.express as px

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
