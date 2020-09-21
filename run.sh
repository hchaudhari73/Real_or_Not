echo $1

if [ $1 = "model" ]; then 
    python model.py; 
elif [ $1 = "lm" ]; then
    python log_model.py
elif [ $1 = "pred" ]; then
    python prediction.py
elif [ $1 = "dl" ]; then
    python keras_model.py
elif [ $1 = "ep" ]; then
    python embedding_pred.py
fi
