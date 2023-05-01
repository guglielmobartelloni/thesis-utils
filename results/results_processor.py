
from pathlib import Path

with open('./adfa_2023_05_01_03_02_pm.txt', 'r') as f:
    for line in f.read().strip().split('\n'):
        line = line.split(' ')
        metric = line[0]
        model = Path(line[2]).stem
        mode = model.split('_')[1]
        print(model)

