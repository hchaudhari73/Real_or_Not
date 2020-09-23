import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

TEST_PATH = "test.csv"
MODEL_PATH = "embedding_model/"
VECTORIZOR_PATH = "vectorizors/keras_tokenizer.pkl"
SUBMISSION_PATH = "submissions/embedding_csv"

# model variables
max_len = 100
padding = "post"
trunc = "post"

# loading tokenizer
token_in = open(VECTORIZOR_PATH, "rb")
tokenizer = pickle.load(token_in)

#load model
model = keras.models.load_model(MODEL_PATH)

# loading test data
test = pd.read_csv(TEST_PATH)
X_test = test[["keyword", "text"]]

# creating corpus for vectorization
test_corpus = [str(i)+" "+str(j) for i,
               j in zip(X_test["keyword"], X_test["text"])]

X_test_seq = tokenizer.texts_to_sequences(test_corpus)
X_test_pad = np.array(pad_sequences(
    X_test_seq, maxlen=max_len, padding=padding, truncating=trunc))

#predicting test data
test["target"] = model.predict(X_test_pad)
test["target"] = test["target"].map(lambda x: round(x))

#creating submission file
submission = test[["id", "target"]]

#saving submission
submission.to_csv(SUBMISSION_PATH, index=False)