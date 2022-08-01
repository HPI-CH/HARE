
import json

metrics = ["Accuracy", "F1 Macro", "F1 Weighted"]
configs = ['SENSOR - x', 'SENSOR - POSE', 'x - POSE']

data = {}
for conf in configs:
    with open(f"Data {conf}.json") as file:
        dict = json.load(file)
        data[conf] = {metric: dict[metric] for metric in metrics}

cell_len = 17

print('\n', ' '*cell_len, end='')
for conf in configs:
    print(conf, ' '*(cell_len - len(conf)), end='')
print('')
for metric in metrics:
    print('\n', metric, ' '*(cell_len - len(metric) - 1), end='')
    
    for conf in configs:
        value = "{:1.5f}".format(data[conf][metric]) 
        print(value, ' '*(cell_len - len(value)), end='')

input("")

