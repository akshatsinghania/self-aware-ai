class Node:
    def __init__(self, info, excite, values):
        self.info = info
        self.excite = excite
        self.value = sum(values) / len(values)

layers=[[Node()]]
