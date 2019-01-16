# Disaster Response Pipeline Project

### 1. Installation
You need install **nltk**, **sklearn**, **plotly**  libraries. Besides, no need to install additional libraries other than Python integrated in Anaconada.The code should run using Python 3.* .
  
### 2. Project Motivation
In this project I analyze disaster data from [Fieure Eight](https://www.figure-eight.com/) to build a machine model to classify disaster messages using Python scripts:
1. Write a data cleaning pipeline. 
2. Write a machine learning pipeline.
3. Add two exrta features in the Flask Web App provided by Udacity.

### 3.File Descriptions
There are 3 Python scrips that have be witten or changed, Also, these scrips output a databse and a pickle file.
1. `data/process_data.py `  
* Load the `disaster_messages.csv` and `disaster_categories.csv`. Second, 
* Merget the two datasets. 
* Clean the data.
* Store it in a SQLite databse, for example `DisasterResonse.db`   
2. ` models/train_classifier.py `
* Load the data from the SQLite database built by `data/process_data.py`, for example `DisasterResonse.db` .
* Splits the dataset into training and test sets. 
* Build a machine pipeline of ` CountVectorizer`, `TfidfTransformer` and `MultiOutputClassifier(RandomForestClassifier)`.
* Use the `GridSearchCV` to find the best parameters
* Output result on the test set using f1 score, precision and recall.
* Export the final model as a pickle file.
3. `app/run.py`
   In this Python script, I add two extra visuals showing in thr Flask web app.
### 4. Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/  
   
### 5. Licensing, Authors, and Aknowledgements
Must give credit to [Fieure Eight](https://www.figure-eight.com/) for the data.You can find the Licensing for the data and other descriptiv information at Figure Eight in available [here](https://www.figure-eight.com/legal/). Also Must give credit to Udacity for the code and the project template. Otherwise, feel free to use the code here as you would like. This project was committed to the GitHub [here](https://github.com/caoconghan/DataScientist_DiasterResponsePipeline.git), everyone can download or use as you would like.

