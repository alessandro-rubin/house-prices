import os
import pandas as pd
import sys


def main():
    cur_path = os.path.dirname(__file__)
    print(cur_path,'\n')
    relpath=os.path.relpath(os.path.dirname(__file__))#can also use realpath instead
    print(relpath)
    train_path=os.path.join(cur_path,'train.csv')
    print(train_path)
    train=pd.read_csv(train_path)
    print(train.head())

if __name__ == '__main__':
    main()