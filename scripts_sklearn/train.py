import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score,make_scorer,mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
import pathlib
from utils import *

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

def pipeline_builder(n_cols,c_cols,o_cols,b_cols):
    numeric_transformer = Pipeline(steps=[
        ('inp',SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    binned_transformer = Pipeline(steps=[('inp',SimpleImputer(strategy='median')),
                                    ('KBinsDiscretizer',KBinsDiscretizer(n_bins=10))])
    ordinal_transformer = Pipeline(steps=[('onehot', OrdinalEncoder())])
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, n_cols),
        ('cat', categorical_transformer, c_cols),
        ('ord', ordinal_transformer, o_cols),
        ('bin',binned_transformer, b_cols)
        ])
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', GradientBoostingRegressor())])
    pipeline
    return pipeline



def load_data():
    return 0

def xy_split(df,label):
    x,y=df.drop(label,axis=1),df[label]
    return x,y


def make_prediction(model,test_df):
    pred=model.predict(test_df)
    out_df=test_df['Id']
    out_df['SalePrice']=pred
    return out_df

def main():
    cur_path = os.path.dirname(__file__)
    print(cur_path,'\n')
    #relpath=os.path.relpath(os.path.dirname(__file__))#can also use realpath instead
    main_path=pathlib.Path(cur_path).parent
    print(main_path)
    train_path=main_path/'data'/'train.csv'
    test_path=main_path/'data'/'train.csv'

    train,test=pd.read_csv(train_path),pd.read_csv(test_path)

    print('Train and test datasets loaded.')

    train,test=feat_eng(train),feat_eng(test)


    label='SalePrice'

    int_cat_f=['SaleCondition','TotRmsAbvGrd','ExterQual','Exterior1st','Neighborhood',
            'GarageFinish','KitchenQual','SaleType','PoolQC', 'OverallCond',
            'FullBath','HouseStyle','Condition1','MSZoning','BldgType','BsmtQual']
    int_num_f=['LotArea','LotFrontage','BsmtFinSF1','TotalBsmtSF',
            'GrLivArea','GarageYrBlt','GarageArea','YearBuilt','MSSubClass']
    int_ord_f=['GarageCars','OverallQual']

    binned_features=[]
    pipeline=pipeline_builder(int_num_f,int_cat_f,int_ord_f,binned_features)

    # define the parameters or load them from configuration file
    params = [{'regressor': [GradientBoostingRegressor()],
    'regressor__learning_rate': [0.03,0.1, 0.5, 1.0],
    'regressor__n_estimators' : [25,50, 100, 150,200]
    },
    {
        'regressor':[RandomForestRegressor()],
        'regressor__n_estimators' : [25,50, 100, 150,200]
    },
    {
    'regressor':[AdaBoostRegressor()],
    'regressor__learning_rate':[0.03,0.1, 0.5, 1.0],
    'regressor__n_estimators' : [25,50, 100, 150,200],
    'regressor__loss':['linear','square']
    },
    {
    'regressor':[XGBRegressor()],
    'regressor__gamma':[0,1,10,1000],
    'regressor__eta':[0.1,0.2,0.3,0.4],
    }
    ]


    x,y=xy_split(train,label)
    score = make_scorer(r2_score)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    gridsearch=GridSearchCV(verbose=1,estimator=pipeline,param_grid=params,scoring=score,return_train_score=False,cv=3)
    model1=gridsearch.fit(x_train,(y_train))

    print('GridCV performed, best parameters: ')
    print(model1.best_params_)

    #model1.fit()


    from sklearn.metrics import mean_squared_log_error
    predictions=model1.predict(x_test)
    msle=mean_squared_log_error(y_true=y_test, y_pred=predictions)#msle is the metric evaluated in the kaggle challange
    r2=r2_score(y_true=y_test, y_pred=predictions)

    print(f'r2 score: {r2:.4f}'
    f'msle: {msle:.1f}')

    savemodel=True
    if savemodel:
        path=main_path/'models'
        save_model(model1,path)
    print('DONE.')

if __name__ == '__main__':
    main()
