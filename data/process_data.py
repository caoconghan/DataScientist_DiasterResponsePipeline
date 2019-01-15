import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """load data and merge data from two filepath.The file MUST be csv format

    Keyword arguments:
    messages_filepath -- string, file path
    categories_filepath -- string, file path
    """
    #Read datas from the messages.csv to DataFrame
    messages = pd.read_csv(messages_filepath)
    #Read datas from the categories.csv to DataFrame
    categories = pd.read_csv(categories_filepath)
    #Merge to DataFrame
    df = messages.merge( categories, on='id')
    
    return df

def clean_data(df):
    """Clean the data
     Keyword arguments:
     df -- DataFrame
    """
    #Split categories into separate category columns.
    categories = df['categories'].str.split(';', expand=True)    
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0]) 
    category_colnames = category_colnames.apply(lambda x: x.replace('_', ' '))    
    categories.columns = category_colnames
     
    #Convert category values to just numbers 0 or 1
    for column in categories:       
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]    
        # convert column from string to numeric
        categories[column] = categories[column].astype(str).astype(int)   
    #Replace categories column in df with new category columns.               
    df =df.drop(['categories'], axis=1)
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """Save the clean dataset into an sqlite database.
     Keyword arguments:
     df -- DataFrame
     database_filename -- string
    """
    #get the tablename from databese_filename
    tablename = database_filename.split('/')[-1]
    tablename = tablename.split('.')[-2]
    #create sqlite engine by database_filename
    engine = create_engine('sqlite:///{}'.format(database_filename))
    
    #Create and write sql table
    df.to_sql(tablename, engine, index=False, if_exists = 'replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()