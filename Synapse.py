import numpy as np

synapse_size = 4
synapse_hosts = np.arange(1, synapse_size + 1)
synapse_target = np.arange(2, synapse_size + 1)
synapse_target = np.append(synapse_target, 1)

print("Synapse size", 4)
print("Synapse Host", synapse_hosts)
print("Synapse Target", synapse_target)
