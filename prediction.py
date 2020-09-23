import pandas as pd 
import pickle

TEST_PATH = "data/test.csv"
MODEL_PATH = "models/log_model.pkl"
VECTORIZOR_PATH = "vectorizors/vectorizor.pkl"

#loading trained model
pickle_out = open(MODEL_PATH, "rb")
MODEL = pickle.load(pickle_out)

#loading vectorizor
vec_out = open(VECTORIZOR_PATH, "rb")
VEC = pickle.load(vec_out)

#reading test data
test = pd.read_csv(TEST_PATH)

#modifying test data as per trained data
X_test = test[["keyword", "text"]]

#creating corpus for vectorization
corpus = [str(i)+" "+str(j) for i, j in zip(X_test["keyword"], X_test["text"])]

X_test_vec = VEC.transform(corpus)

test["target"] = MODEL.predict(X_test_vec)

submission = test[["id", "target"]]

submission.to_csv("submissions/submission.csv", index=False)

