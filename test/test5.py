import numpy as np

emb = np.load("../embeddings2.npy")

emb = np.append(emb, [0,1,2,3,4,5,6,7])

print(emb.size)