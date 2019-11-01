import numpy as np
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
np.random.seed(2019)

def gen_patterns(x, pos, m, patterns):
    if pos == m:
        patterns.append(np.array(x))
        return patterns
    
    for i in [-1, 1]:
        x[pos] = i
        patterns = gen_patterns(x, pos + 1, m, patterns)
    
    return patterns

def spurious_search(patterns, samples, weights):
    spurious = []
    
    for z in patterns:
        # Check that pattern is spurious
        if (not any([np.array_equal(z, y) for y in samples]) and
                not any([np.array_equal(-z, y) for y in samples]) and
                np.array_equal(z, np.sign(weights @ z))):
            spurious.append(np.array(z))
        
    return spurious

n = 5
m = 4

# Generate patterns
patterns = gen_patterns(np.zeros((m, 1)), 0, m, [])

# Extract some independent samples
samples = []
shuffled_samples = np.random.permutation(patterns)

i = 0
while len(samples) < n:
    if (not any([np.array_equal(shuffled_samples[i], y) for y in samples]) and
            not any([np.array_equal(-shuffled_samples[i], y) for y in samples])):
        samples.append(patterns[i])
    i = i + 1

print("Selected patterns:")
for x in samples:
    print(x.transpose())

# Compute weights
weights = sum([x @ x.transpose() for x in samples])
print("Weight matrix:")
print(weights)

# Retrieve spurious patterns
spurious = spurious_search(patterns, samples, weights)
print("Spurious patterns:")
for x in spurious:
    print(x.transpose())