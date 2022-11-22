# 

import os
import pandas as pd
import sys

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

def main():
    cur_path = os.path.dirname(__file__)
    print(cur_path,'\n')
    #relpath=os.path.relpath(os.path.dirname(__file__))#can also use realpath instead
    train_path=os.path.join(cur_path,'train.csv')
    test_path=os.path.join(cur_path,'test.csv')
    train,test=pd.read_csv(train_path),pd.read_csv(test_path)
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


if __name__ == '__main__':
    main()