import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import ktrain

TRAIN_PATH = "data/clean_train.csv"
TEST_PATH = "data/test.csv"


# model parameters
max_len = 100
mode = "bert"

# reading train data
df = pd.read_csv(TRAIN_PATH)
df.head()

# training size
TRAIN_SIZE = int(0.8*len(df))

# seperating train and validation set
train = df[["stemmed_text", "target"]][:TRAIN_SIZE]
val = df[["stemmed_text", "target"]][TRAIN_SIZE:]

# train-test split
(X_train, y_train), (X_test, y_test), preproc = ktrain.text.texts_from_df(
    train_df=train,
    text_column="stemmed_text",
    label_columns="target",
    val_df=val,
    maxlen=max_len,
    preprocess_mode=mode
)

# creating bert model
model = ktrain.text.text_classifier(
    name=mode,
    train_data=(X_train, y_train),
    preproc=preproc
)

learner = ktrain.get_learner(
    model=model,
    train_data=(X_train, y_train),
    val_data=(X_test, y_test),
    batch_size=3
)

# training model
learner.fit_onecycle(lr=2e-5, epochs=1)
predictor = ktrain.get_predictor(learner.model, preproc)

#reading test data
test = pd.read_csv(TEST_PATH)
test.head()

#only considering one column, text.
X_test = test["text"].to_list()
test["target"] = predictor.predict(X_test)

#converting target column from object to numeric
test["target"] = test["target"].map({
    "not_target": 0,
    "target": 1
})

#creating and saving submission file
submission = test[["id", "target"]]
submission.to_csv("submission/bert_text_submission.csv", index=False)

#saving model
predictor.save("models/bert_text")
