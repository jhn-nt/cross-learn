from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from typing import List, Iterable


class ElbowKMeans(KMeans):
    def __init__(self,k:Iterable[int],**kwargs):
        pass



