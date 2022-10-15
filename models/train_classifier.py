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
nltk.download(['wordnet', 'punkt', 'stopwords'])

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    table_name=(engine.table_names())
    df = pd.read_sql_table(table_name[0],engine)
    X = df.message
    y = df.iloc[:,5:]
    return X,y
    
    pass


def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    
    words = word_tokenize(text)
    stop = stopwords.words("english")
    words = [t for t in words if t not in stop]
    
    # Lemmatization
    lemm = [WordNetLemmatizer().lemmatize(w) for w in words]
    return lemm


def build_model():
    '''Building a machine learning pipeline for the model'''
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    parameters =  {
    'clf__estimator__n_estimators': [10,12,15],
    'clf__estimator__min_samples_split': [2, 5, 6],
    'clf__estimator__random_state':[42]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    cv.fit(X_train, y_train)
    model=cv.best_estimator_
    
    return model


def evaluate_model(model, X_test, Y_test):
    y_pred = model.predict(X_test)
    for i,value in enumerate(Y_test):
        print('category_name:',Y_test.columns[i])
        print(classification_report(Y_test[value], y_pred[:,i]), '...................................................')
    accuracy = (y_pred == Y_test.values).mean()
    print("Accuracy:", accuracy)


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


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
        evaluate_model(model, X_test, Y_test)

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