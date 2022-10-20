# import required libraries
import sys
import re
import pickle
import pandas as pd 
from sqlalchemy import create_engine
import nltk 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,  GridSearchCV 
from sklearn.metrics import classification_report
nltk.download((['wordnet', 'punkt', 'stopwords','omw-1.4']))

def load_data(database_filepath):
    """Load database and define feature and target variables.
    Args:
    database_filepath: path of dataset containing sqlite database.
    Returns: 
    X: Disaster messages.
    y: Category dataset of disaster messages
    category_names: list of category names"""
    engine = create_engine('sqlite:///'+database_filepath)
    # read table name from database
    table_name=(engine.table_names())
    # load dataset from database
    df = pd.read_sql_table(table_name[0],engine)
    X = df.message
    y = df.iloc[:,4:]
    category_names = y.columns
    return X,y,category_names
    
    pass


def tokenize(text):
    """A tokenisation function to process messages"""
    # removing numbers, special characters and transforming text to lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    words = word_tokenize(text)
    stop = stopwords.words("english")
    # removing stopwords
    words = [t for t in words if t not in stop]
    # lemmatization
    lemm = [WordNetLemmatizer().lemmatize(w) for w in words]
    return lemm


def build_model():
    '''Building a machine learning pipeline for the model'''
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    # initializing parameters for the gridsearch
    parameters =  {
    'clf__estimator__n_estimators': [10],
    'clf__estimator__min_samples_split': [2],
    'clf__estimator__random_state':[42]
    }
    # Assigning the pipeline model
    model=GridSearchCV(pipeline, param_grid=parameters)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """evaluate the f1 score, precision and recall for each output category of the dataset"""
    # Predict test labels
    y_pred = model.predict(X_test)
    # Generating classification report
    for i,value in enumerate(Y_test):
        print('category_name:',Y_test.columns[i])
        print(classification_report(Y_test[value], y_pred[:,i]), '...................................................')
    # Calculating accuracy of model
    accuracy = (y_pred == Y_test.values).mean()
    print("Accuracy:", accuracy)


def save_model(model, model_filepath):
    """Export model as a pickle file"""
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        # load data and perform train text split
        X, Y,category_names = load_data(database_filepath)
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