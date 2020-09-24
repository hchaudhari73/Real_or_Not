# Real or Not

<h2>  NLP with Disaster Tweets Predict which Tweets are about real disasters and which ones are not. </h2>

<h3> First kaggle competition </h3>
<h4> Did something new. </h4>
<p> Transfer learning: Using pretrained model, BERT for prediction. </br> 
    Ktrain: Keras is the only deeplearning framework I know how to use and could find bert model for it. So used "Ktrain" library for it. </br> 
    Used git as it is supposed to: Before this, I just used to drag and drop. Now I know how to do it in terminal. </br>  
    </p>

<h4> Boilerplate stuff for NLP </h4>
<p> Regex: Extracting hashtags, mentions and links(saved in special.txt) </br> 
    Cleaning data: Removing puctuations, stemming tokens(code is available in data_preprocessing.py). </br> 
    </p>
    
<h4> Step-by-Step procedure </h4>
<p> Metric: f1-score </p>
<p> Used multiple models and check which gives better results(code in model.py, dependent on clas.py and results in combine_results.csv). </br> 
    Logistic Regression preformed well. So tested with that(code in log_model.py, prediction.py, model in log_model.pkl). </br> 
    Embedding model: Used simple deep learning model using keras embedding layer(code in keras_model.py, emdedding_pred.py, model in embedding_model). </br>
    BERT model: Using ktrain library to use pretrained model(code in bert_ktrain.py, model in bert_ktrain). </br>
    BERT model: Same as above but only on text column(code in bert_text.py, model in bert_text).
   </p>

<h4> Content </h4>  
<p> data </br>  - train.csv: training data </br>   
                - test.csv: test data </br>
                - clean_train.csv: cleaned data after removing hastags, mentions and links and stemming tokens. </br>
                </p>
                
<p> models </br> - Trained models </br> 
                   </p>
            
<p> submissions </br> - submission files for the competition  </br> 
                </p>     
                
<p> vectorizors </br> - vectorizors trained while modelling. </br> 
                </p>     
                
<h6>Highlights </br> -- Tensorflow </br>
                     -- keras </br>   
                     -- Deep Learning Alogrithms </br> 
                     -- Tranfer learning using BERT and Ktrain </br>
</h6>
