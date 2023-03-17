# 

import os
import pandas as pd
import sys
import numpy as np
from  model_builder import *
import pathlib

def count_na_per_col(df):
    import pandas as pd
    count=df.isna().sum()
    count=pd.DataFrame(count)
    count=count[count[0]!=0]
    return count

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





def train_val(df:pd.DataFrame,val_fraction,feature_list,label,shuffle=False):
    '''returns the train and validation sets as dictionaries'''
    if shuffle==True:
        df=df.sample(frac=1)
    
    val_size=int(val_fraction*len(df))
    train_size=len(df)-val_size
    print('train size = ', train_size, '\nvalidation size = ', val_size)
    train_df=df.iloc[:train_size]
    val_df=df.iloc[train_size:]
    train_dict={name:np.array(train_df[name]) for name in feature_list}
    val_dict={name:np.array(val_df[name]) for name in feature_list}
    train_label=train_df[label]
    val_label=val_df[label]
    return train_dict,train_label,val_dict,val_label





def main():
    print('CUDA check: \n')
    print(tf.test.is_built_with_cuda())

    cur_path = os.path.dirname(__file__)
    print(cur_path,'\n')
    #relpath=os.path.relpath(os.path.dirname(__file__))#can also use realpath instead
    main_path=pathlib.Path(cur_path).parent
    print(main_path)
    train_path=main_path/'data'/'train.csv'
    test_path=main_path/'data'/'test.csv'

    train,test=pd.read_csv(train_path),pd.read_csv(test_path)
    train.drop([523,1298]) ##dropping bad rows

    print('Train and test datasets loaded.')
    train,test=feat_eng(train),feat_eng(test)
    print(train.head())



    int_cat_f=['SaleCondition','ExterQual','Neighborhood','KitchenQual','SaleType','PoolQC','MSZoning','GarageQual','Condition1','BldgType','FireplaceQu','GarageType']
    int_num_f=['OverallQual','LotArea','BsmtFinSF1','TotalBsmtSF','GrLivArea','GarageArea','YearBuilt','GarageCars','TotRmsAbvGrd','MSSubClass','WoodDeckSF']
    label='SalePrice'

    split_fraction=0.2


    inputs = {}
    for name in int_num_f+int_cat_f:
        if name in int_num_f:
            dtype = tf.float32
        if (name in int_cat_f):
            dtype = tf.string
        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)
    

    house_prices_preprocessing=create_preprocessing_layer(int_num_f,int_cat_f,train)
    house_price_model = create_model(house_prices_preprocessing, inputs)

    train_features_dict={name:np.array(train[name]) for name in (int_num_f+int_cat_f)}

    history2=house_price_model.fit(x=train_features_dict, y=train[label], epochs=200,batch_size=15,validation_split=split_fraction)

if __name__ == '__main__':
    main()