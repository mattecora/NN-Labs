import numpy as np
import matplotlib.pyplot as plt

def energy(x, W, b):
    return (- x.transpose() @ W @ x - 2 * x.transpose() @ b).item()

def sync_update(W, b):
    # Generate inputs
    for x0 in [np.array([a, b, c]).reshape(3, 1) for a in [-1, 1] for b in [-1, 1] for c in [-1, 1]]:
        # Compute successor state
        x1 = np.array(np.sign(W @ x0 + b), dtype="int")
        
        # Print states and energies
        print(f"{x0.transpose()} : {energy(x0, W, b)}")
        print(f" -> {x1.transpose()} : {energy(x1, W, b)}")

def async_update(W, b):
    # Generate inputs
    for x0 in [np.array([a, b, c]).reshape(3, 1) for a in [-1, 1] for b in [-1, 1] for c in [-1, 1]]:
        x1i = []

        for j in range(3):
            x1 = np.copy(x0)
            x1[j] = np.sign(W[j] @ x0 + b[j])
            x1i.append(x1)
        
        print(f"{x0.transpose()} : {energy(x0, W, b)}")
        for j in range(3):
            print(f" -> ({j}) : {x1i[j].transpose()} : {energy(x1i[j], W, b)}")

# Define vectors
b = np.array([0.5, 0.5, 0.5]).reshape(3, 1)
W = np.array([0, 2, -1, 2, 0, -1, -1, -1, 1]).reshape(3, 3)
limit = 10

print("### SYNCHRONOUS UPDATE ###")
sync_update(W, b)
print()
print("### ASYNCHRONOUS UPDATE ###")
async_update(W, b)