import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import pickle

TRAIN_PATH = "clean_train.csv"
MODEL_PATH = "models/log_model.pkl"
VECTORIZOR_PATH = "vectorizors/vectorizor.pkl"

# read data
df = pd.read_csv(TRAIN_PATH)

# training size
TRAIN_SIZE = int(0.8*len(df))

# seperating dependent independent features
X = df[["keyword", "stemmed_text"]]
y = df[["target"]]

#train-test split
X_train, y_train = X[:TRAIN_SIZE], y[:TRAIN_SIZE]
X_test, y_test = X[TRAIN_SIZE:], y[TRAIN_SIZE:]

# creating corpus for vectorization
train_corpus = [i+" "+j for i,
                j in zip(X_train["keyword"], X_train["stemmed_text"])]
test_corpus = [i+" "+j for i,
               j in zip(X_test["keyword"], X_test["stemmed_text"])]

# verctorizion
vec = TfidfVectorizer()

X_train_vec = vec.fit_transform(train_corpus)
X_test_vec = vec.transform(test_corpus)

#model
model = LogisticRegression()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
score = round(f1_score(y_pred, y_test), 3)
print(f"f1-score: {score}")

# saving model 
with open(MODEL_PATH, "wb") as pickle_out:
    pickle.dump(model, pickle_out)
    print("model saved")

#saving vectorizor
with open(VECTORIZOR_PATH, "wb") as vec_out:
    pickle.dump(vec, vec_out)
    print("vec saved")