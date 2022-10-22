# Disaster Response Pipeline Project
This project analyze disaster data from Appen (formally Figure 8) to build a model for an API that classifies disaster messages.

In the Project Workspace, There is a data set containing real messages that were sent during disaster events. A  machine learning pipeline is created to categorize these events so that you can send the messages to an appropriate disaster relief agency.

This project  includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

### Project Components
There are three components in this project.

1. ETL Pipeline
Python script,`process_data.py `is a data cleaning pipeline that:

Loads the messages `disaster_messages.csv ` and categories `disaster_categories.csv ` datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database

2. ML Pipeline
Python script,`train_classifier.py ` is a machine learning pipeline that:

Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file

3. Flask Web App
Python script,`run.py ` initiates a flask web app that:

Loads data from the SQLite database
Extract required data for visualizations
Launch a web app where an emergency worker can input a new message and get classification results in several categories
Visualizing data using Plotly

### Tree of folders and files
``` 
DisasterResponse
│   .gitattributes # attributes to pathnames
│   README.md # Information and instructions about project
│   requirements.txt # required libraries to install
│
├───app
│   │   run.py # Flask file that runs app
│   │   Screenshot_web_app1.png # screenshot of web app
│   │   Screenshot_web_app2.png # screenshot of web app
│   │   Screenshot_web_app3.png # screenshot of web app
│   │
│   └───templates
│           go.html # classification result page of web app
│           master.html # main page of web app
│
├───data
│       DisasterResponse.db # database to save cleaned data
│       disaster_categories.csv # data to process
│       disaster_messages.csv # data to process
│       process_data.py # data cleaning pipeline
│       
│
└───models
        classifier.pkl # saved model
        train_classifier.py # ML pipeline
```
### Instructions:
1. Install the required libraries in 'requirements.txt' file from `DisasterResponse` directory
1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
      Go to `data` directory: `cd data` and input the below line in terminal
        `python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
      Go to `models` directory: `cd models` and input the below line in terminal
        `python train_classifier.py ../data/DisasterResponse.db classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run web app: `python run.py`

