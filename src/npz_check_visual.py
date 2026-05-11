import matplotlib.pyplot as plt
import numpy as np

data = np.load("extracted_features/bin/bbaf2n_23_29.npz", allow_pickle=True)
frame = data["normalized"][0].reshape(40, 2)  # first frame, 40 (x,y) pairs

plt.figure(figsize=(4, 3))
plt.scatter(frame[:, 0], frame[:, 1], s=20)
plt.gca().invert_yaxis()  # image coordinates: y increases downward
plt.title("Lip landmarks - frame 0 (normalized)")
plt.xlabel("x"); plt.ylabel("y")
plt.tight_layout()
plt.show()