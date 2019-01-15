import sys
import nltk
import sqlite3
from sqlalchemy import create_engine

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.externals import joblib

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    """Load data from sqlite: database_filepath
    Keyword arguments:
    database_filepath -- string,  sqlite Database file path
    """
    sqlite_database_filepath = 'sqlite:///{}'.format(database_filepath.replace('../',''))
    
    engine = create_engine(sqlite_database_filepath)
    database_tablename = sqlite_database_filepath.split('/')[-1][:-3]
          
    df = pd.read_sql_table(database_tablename, engine)
              
    X = df.message.values
    Y = df.iloc[:, 4:].values
    category_names = df.columns[4:]
             
    return X, Y, category_names              


def tokenize(text):
    """A tokenization function to process your text data
    """
    #stopwords, lemmatizer
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
              
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ",text)
    
    # tokenize text
    tokens = word_tokenize(text)
              
    # lemmatize andremove stop words
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok, pos='v').lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    """Build a machine learning pipeline and Grid Search
    """
    #Pipeline includes CountVectorizer,TfidfTransformer and RandomForestClassifier
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, ngram_range =  (1,2), max_df = 0.5, max_features = 5000)),
        ('tfidf', TfidfTransformer(use_idf = True)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 200, min_samples_split = 3)))
    ])
     
    #The Grid Search parameters
    parameters = {   
        #'vect__ngram_range': ((1,1), (1,2)),
        #'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__max_features': (None, 5000, 10000),
        #'tfidf__use_idf': (True, False),
        #'clf__estimator__n_estimators':[50, 100, 200],
        #'clf__estimator__min_samples_split': [2,3,4]
        'clf__estimator__min_samples_split': [3,4]
        }
    #Grid Search
    model = GridSearchCV(pipeline, param_grid=parameters, verbose = 2)
    #model = pipeline
    
    return model            
            
              
def evaluate_model(model, X_test, Y_test, category_names): 
    """Test Model with the f1 score, precision and recall for each output category of the dataset
     Keyword arguments:
     model -- a machine learning model
     X_test ,Ytest -- Test datasets
     category_names -- category names dataset
     """
    # predict on test data
    Y_pred = model.predict(X_test)
    
    #Report f1 score, precision and recall for each output categoty 
    for i in range(Y_test.shape[1]):
        report = classification_report(Y_test[:,i], Y_pred[:,i])
        print("{}: \n".format(category_names[i]))
        print(report)
          


def save_model(model, model_filepath):
    """Save the model as a pickle file on mode_filepath
    """
    #
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
        
        
             

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()