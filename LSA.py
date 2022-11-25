import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

def lsa_calc(n):
    svd = TruncatedSVD(n_components=n)
    lsa = svd.fit_transform(bag_of_words)
    return lsa
