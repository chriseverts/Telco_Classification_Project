import pandas as pd
import os
from env import username, host, password 
from sklearn.model_selection import train_test_split

def train_validate_test_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test

# Acquiring telco_churn data
def get_connection(db, username=username, host=host, password=password):
    '''
    Creates a connection URL
    '''
    return f'mysql+pymysql://{username}:{password}@{host}/{db}'


def new_telco_churn_data():
    '''
    Returns telco_churn into a dataframe
    '''
    sql_query = '''select * from customers
    join internet_service_types using(internet_service_type_id)
    join contract_types using(contract_type_id)
    join payment_types using(payment_type_id)'''
    df = pd.read_sql(sql_query, get_connection('telco_churn'))
    return df 


def get_telco_churn_data():
    '''get connection, returns telco_churn into a dataframe and creates a csv for us'''
    if os.path.isfile('telco_churn.csv'):
        df = pd.read_csv('telco_churn.csv', index_col=0)
    else:
        df = new_telco_churn_data()
        df.to_csv('telco_churn.csv')
    return df
    