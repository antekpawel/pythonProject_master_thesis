#Generowanie wykresu wspolczynika przeplywu od liczby raynoldsa dla rury
import numpy as np
import plotly.express as px


mi = 1e-3 #[Pas]
ro = 1000 #[kg/m3]
d0 = 0.1 #[m]
vs = np.geomspace(0.0001, 0.1, 100, endpoint=0) #[m/s]
print(vs)

Re = vs * d0 * ro / mi
lam = np.empty_like(Re)

for i in range(len(Re)):
    if Re[i] < 2100:
        lam[i] = 64 / Re[i]
        print("Pzeplyw laminarny")
    elif Re[i] >=2100 and Re[i] < 3e3:
        lam[i] = 0.3164 / pow(Re[i], 0.25)
        print("Przeplyw przejsciowy")
    else:
        lam[i] = 0.0032 + 0.221 / pow(Re[i], 0.237)
        print("Przeplyw turbulentny")
#print(lam)

fig = px.line(vs, lam)
fig.show()


