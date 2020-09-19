import pandas as pd
import numpy as np
from nltk import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from clas import clas

df = pd.read_csv("train.csv")

df.drop()