from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import f1_score
import pickle

TRAIN_PATH = "data/clean_train.csv"
MODEL_PATH = "models/embedding_model"
VECTORIZOR_PATH = "vectorizors/keras_tokenizer.pkl"

# model variables
vocab_size = 2000
oov_token = "<OOV>"
max_len = 100
padding = "post"
trunc = "post"
embedding = 32
EPOCH = 30

# read data
df = pd.read_csv(TRAIN_PATH)

# training size
TRAIN_SIZE = int(0.8*len(df))

# seperating dependent independent features
X = df[["keyword", "stemmed_text"]]
y = df[["target"]]

# train-test split
X_train, y_train = X[:TRAIN_SIZE], y[:TRAIN_SIZE]
X_test, y_test = X[TRAIN_SIZE:], y[TRAIN_SIZE:]

# creating corpus for vectorization
train_corpus = [i+" "+j for i,
                j in zip(X_train["keyword"], X_train["stemmed_text"])]
test_corpus = [i+" "+j for i,
               j in zip(X_test["keyword"], X_test["stemmed_text"])]

# Tokenization
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)

tokenizer.fit_on_texts(train_corpus)
X_train_seq = tokenizer.texts_to_sequences(train_corpus)
X_train_pad = np.array(pad_sequences(
    X_train_seq, maxlen=max_len, padding=padding, truncating=trunc))

X_test_seq = tokenizer.texts_to_sequences(test_corpus)
X_test_pad = np.array(pad_sequences(
    X_test_seq, maxlen=max_len, padding=padding, truncating=trunc))

print(X_train_pad.shape, y_train.shape, X_test_pad.shape, y_test.shape)

# model
keras.backend.clear_session()  # clearing if any session already running

model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding, input_shape=[max_len]),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(6, activation='relu', kernel_initializer="he_uniform"),
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam",
              loss=keras.losses.binary_crossentropy, metrics=["accuracy"])

# model.fit(X_train_pad, y_train, epochs=EPOCH, validation_data=(X_test_pad, y_test))
model.fit(X_train_pad, y_train, epochs=EPOCH)

#for f1-score
y_pred = model.predict(X_test_pad)
#converting predicted probabilites to binar class
y_pred = [np.round(i[0]) for i in y_pred] 
score = f1_score(y_pred, y_test)
print(f"f1-score = {np.round(score, 3)}")

#saving model
model.save(MODEL_PATH)

#saving tokenizer
with open(VECTORIZOR_PATH, "wb") as token_out:
    pickle.dump(tokenizer, token_out)