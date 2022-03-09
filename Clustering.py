from metapath2vec import model
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from numpy import random, where
import pickle as pk

with open('embedding.pkl', 'rb') as f:
    embedding =pk.load(f)

print(embedding)

clustering = DBSCAN(eps=2, min_samples=2).fit(embedding.cpu().detach().numpy())
print(clustering.labels_)