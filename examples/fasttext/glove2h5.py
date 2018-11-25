"""Convert raw GloVe word vector text file to h5."""
import h5py
import numpy as np

glove_vectors = [
    line.strip().split()
    for line in open('glove.840B.300d.txt', 'r')
]
print("vectors read")
vocab = [line[0] for line in glove_vectors]
print("vocab read")
vectors = np.array(
    [[float(val) for val in line[1:]] for line in glove_vectors]
).astype(np.float32)
vocab = '\n'.join(vocab)
print("vectors created")
f = h5py.File('glove.840B.300d.h5', 'w')
f.create_dataset(data=vectors, name='embedding')
f.create_dataset(data=vocab, name='words_flatten')
f.close()
