# house-prices

House prices analysis for the kaggle challange https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

The aim of the conde in this repository is to predict the sale price of the entries in the the test.csv file.

I will use mainly the scikit-learn and TensorFlow/Keras python packages.


The notebook `house-prices-exploration.ipynb` is a preliminary data analysis and visualization of the dataset In which I perform statistical analysis on the data, showing the distribution of the variouse features, their correlation, looking for outlierse, etc.

The notebook `house-prices.ipynb` contains code for a house price prediction based on the package scikit learn using a cross validation search with multiple parameter and regressors.

The notebook `house-prices-tf.ipynb` contains code for a hosue price prediction uwing a neural network approach with the TensorFlow and Keras packages.