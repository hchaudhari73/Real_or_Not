import pandas as pd 
import pickle

#loading trained model
pickle_out = open("log_model.pkl", "rb")
MODEL = pickle.load(pickle_out)

#loading vectorizor
vec_out = open("vectorizor.pkl", "rb")
VEC = pickle.load(vec_out)

#reading test data
test = pd.read_csv("test.csv")

#modifying test data as per trained data
X_test = test[["keyword", "text"]]

#creating corpus for vectorization
corpus = [str(i)+" "+str(j) for i, j in zip(X_test["keyword"], X_test["text"])]

X_test_vec = VEC.transform(corpus)

test["target"] = MODEL.predict(X_test_vec)

submission = test[["id", "target"]]

submission.to_csv("submission.csv", index=False)

