import pandas as pd  # read data
from sklearn.feature_extraction.text import TfidfVectorizer  # strings to numeric
# from sklearn.model_selection import train_test_split  # train-test split
from clas import clas  # classification models

TRAIN_PATH = "data/clean_train.csv"

df = pd.read_csv(TRAIN_PATH)

TRAIN_SIZE = int(0.8*len(df))

# seperating dependent independent features
X = df[["keyword", "stemmed_text"]]
y = df["target"]

# train-test split
X_train, y_train = X[:TRAIN_SIZE], y[:TRAIN_SIZE]
X_test, y_test = X[TRAIN_SIZE:], y[TRAIN_SIZE:]

#creating corpus for vectorization
train_corpus = [i+" "+j for i,j in zip(X_train["keyword"], X_train["stemmed_text"])]
test_corpus = [i+" "+j for i,j in zip(X_test["keyword"], X_test["stemmed_text"])]

# verctorizion
vec = TfidfVectorizer()

X_train_vec = vec.fit_transform(train_corpus)
X_test_vec = vec.transform(test_corpus)

results = clas(X_train_vec, X_test_vec, y_train, y_test)
results.to_csv("combine_results.csv")
