import json
import numpy as np
import matplotlib.pyplot as plt

l1 = None
l2 = None

with open('data/negative_to_positive.json') as f:
    l1 = json.load(f)

with open('data/positive_to_negative.json') as f:
    l2 = json.load(f)

l1 = np.array(l1).T.tolist()
l2 = np.array(l2).T.tolist()

for i in range(len(l1)):
    x1 = l1[i]
    x2 = l2[i]
    x1.extend(x2)

l = np.array(l1)
m = np.mean(l, axis=1)

plt.title("Mean Sentiment Switch Discrim. Loss (Switch at Token 10)")

plt.xlabel("Tokens Generated")
plt.ylabel("Discriminator Loss")

plt.plot(range(0, m.shape[0]), m, '-')
plt.show()
