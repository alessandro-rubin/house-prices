import os
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
from joblib import dump
import datetime
import numpy as np
import matplotlib.pyplot as plt

def feat_eng(df):
    ''' performs dataframe cleaning and basic feature engineering'''
    #total floors
    df['TotFlrSF']=df['1stFlrSF']+df['2ndFlrSF']
    #total number of floors (1 or 2)
    df['nFlrs']=df['2ndFlrSF'].map(lambda x: int(x>0)+1.)#if df['2ndFlrSF']>0 then df['nFlrs']=2
    df['GarageType']=df['GarageType'].fillna('NoGarage')
    df['GarageQual']=df['GarageQual'].fillna('NoGarage')
    df['GarageCond']=df['GarageCond'].fillna('NoGarage')
    df['PoolQC']=df['PoolQC'].fillna('NoPool')
    df['FireplaceQu']=df['FireplaceQu'].fillna('NoFp')
    #garage cars vs house size
    #number of bathrooms vs house size/n bedrooms
    return df

def score_and_visualize(model,x_test,y_test):
    predictions=model.predict(x_test)
    plt.figure(figsize=(10,10))
    plt.scatter(y_test,predictions)
    plt.xlabel('True')
    plt.ylabel('Predicted')
    msle=mean_squared_log_error(y_true=y_test, y_pred=predictions)#msle is the metric evaluated in the kaggle challange
    r2=r2_score(y_true=y_test, y_pred=predictions)
    print('r2 coefficient: ',r2,'\nmsle = ', msle)
    z = np.polyfit(y_test, predictions, 1)
    p = np.poly1d(z)
    plt.plot(y_test,p(y_test), color='magenta')
    plt.axline((0,0),slope=1,ls='--')
    plt.show()

def save_model(model,models_folder,timestamp):
    '''model: sklearn model
    models_folder: folder where saving the model
    timestamp: timestamp'''
    
    #timestamp=datetime.datetime.now().strftime("%d%m%Y%H%M%S")
    #I have to go up one folder
    os.makedirs(models_folder,exist_ok=True)
    model_name= 'skl_model_'+ timestamp +'.joblib'
    modelpath= os.path.join(models_folder,model_name)
    
    dump(model, modelpath)
    print('Saved ' , modelpath)

def make_prediction(model,test_df):
    pred=model.predict(test_df)
    out_df=test_df['Id']
    out_df['SalePrice']=pred
    return out_df