import numpy as np

example =  [[2, 0, 0, 0, 0],
            [4, 9, 0, 0, 0],
            [4, 9, 6, 0, 0],
            [7, 7, 8, 9, 0],
            [3, 4, 1, 1, 7]]

row = np.amax(np.shape(example))
print(row)
way = np.zeros([2**row, row])

dzban1 = 0
dzban2 = 0

for i in range(2**row):
    for j in range(row):
        way[i][j] = example[dzban1][dzban2]
        dzban1 = dzban1 + 1
        if dzban1 == row:
            dzban1 = 0;



print(way)