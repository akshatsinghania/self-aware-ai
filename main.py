from random import random


class Node:
    def __init__(self, info, excite, values):
        self.info = info
        self.excite = excite
        self.value = self.getValue(values)

    def getValue(self, values):
        if (len(values) == 0):
            return random()
        sum = 0
        for i in values:
            sum += i.value
        return sum / len(values)


last_layer = []
layers_number = 20
layers = []
appeared = {}
for i in range(1, layers_number):
    current_layer = []
    if round(layers_number / (2 * i)) in appeared:
        break
    appeared[round(layers_number / (2 * i))] = 1
    for j in range(1, round(layers_number / (2 * i)) + 2):
        last=0
        offset=1
        if i>1:
            last = (2 * (i - 1))
            offset=round(layers_number/(j+last))
        previous = offset - 1
        nextt = offset + 1
        print(previous,nextt,end='\t\t')

        if i > 1:
            current_layer.append(Node(random(), random(), [last_layer[previous], last_layer[nextt]]))
        else:
            current_layer.append(Node(random(), random(), []))
    layers.append(current_layer)
    last_layer = layers[len(layers) - 1]
    print()

# for x in layers:
#     for y in x:
#         print(round(y.value,ndigits=3),end='\t')
#     print('\n')
