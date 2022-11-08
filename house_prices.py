import os
import pandas as pd
import sys

def feat_eng(df):
    #total floors
    df['TotFlrSF']=df['1stFlrSF']+df['2ndFlrSF']
    #total number of floors (1 or 2)
    df['nFlrs']=df['2ndFlrSF'].map(lambda x: int(x>0)+1.)#if df['2ndFlrSF']>0 then df['nFlrs']=2
    df['AvgSF']=df['TotFlrSF']/df['nFlrs']
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

if __name__ == '__main__':
    main()