import json
import plotly
import pandas as pd


from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar,Pie
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
table_name=(engine.table_names())
df = pd.read_sql_table('DisasterResponse',engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # Message count across genre
    genre_counts = df.groupby('genre').count()['message']
    # Genre count across message categories
    direct_counts = df[df['genre']=='direct'].sum()[df.columns[4:]]
    news_counts = df[df['genre']=='news'].sum()[df.columns[4:]]
    social_counts = df[df['genre']=='social'].sum()[df.columns[4:]]
    
    #Category labels
    genre_names = list(genre_counts.index)
    dir_names =list(direct_counts.index)
    # create visuals
    # 
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
                Pie(
                    labels=dir_names,values=direct_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Direct Genre across message categories',
   
            }
        },
        {
            'data': [
                Pie(
                    labels=dir_names,values=social_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Social Genre across message categories',
   
            }
        },
        {
            'data': [
                Pie(
                    labels=dir_names,values=news_counts
                )
            ],

            'layout': {
                'title': 'Distribution of News Genre across message categories',
   
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
    
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()