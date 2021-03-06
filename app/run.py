import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """tokenize, lemmatize and clean the data

    Keyword arguments:
    test -- string, test to be tokenize
   
    Return :
    clean_tokens -- list,clean tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data from SQLite database
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """add data for visuals

    Keyword arguments: None    
   
    Return :
    render_template--render web page with plotly graphs
    """
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Add the second data for visuals：Distribution of Categories
    category_df= df.iloc[:, 4:]
    category_sum = category_df.sum()
    category_sum = category_sum.sort_values(ascending=False)
    category_counts = category_sum.values
    category_names = list(category_sum.index)
    
    #add the third data for visual ： in the dataset
    category_row_df = df.iloc[:, 4:]
    category_row_df = category_row_df.sum(axis=1)
    
    categoty_describe_names = list(category_row_df.describe().index[1:])
    categoty_describe_values = category_row_df.describe().values[1:]
    describe_names = []
    describe_values = []
    #add describe names
    describe_names.append(categoty_describe_names[0])
    describe_names.append(categoty_describe_names[1])
    describe_names.append(categoty_describe_names[2])
    describe_names.append(categoty_describe_names[6])
    
    #add describe values
    describe_values.append(categoty_describe_values[0])
    describe_values.append(categoty_describe_values[1])
    describe_values.append(categoty_describe_values[2])
    describe_values.append(categoty_describe_values[6])
    
    
    #add 3 graphs to master.html uing dict
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts 
                )
            ],

            'layout': {
                'title': 'Distribution of Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
      {
            'data': [
                Bar(
                    x=describe_names,
                    y=describe_values 
                )
            ],

            'layout': {
                'title': 'Describe of Categories',
                'yaxis': {
                    'title': "Value"
                },
                'xaxis': {
                    'title': "Describe"
                }
            }
        }
       
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """use model to predict classification for query by the go.html

    Keyword arguments: None    
   
    Return :
    render_template--render the go.html
    """
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()