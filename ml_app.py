import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


#  Page title
st.set_page_config(page_title='Chronic Kidney Disease',layout='wide')

data = pd.read_csv('dat.csv')
st.write('Dataset Size')
st.info(data.shape)
st.write('Dataset Columns')
st.info(data.columns)
st.write('Dataset Sample')
st.write(data.head())
ind_col = [col for col in data.columns if col != 'class']
dep_col = 'class'

X = data[ind_col]
y = data[dep_col]
st.write('Splitting data into training and testing')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)
st.write('Shape of our Training data')
st.info(X_train.shape)
st.write('Shape of our Test data')
st.info(X_test.shape)

#st.subheader('Applying Variance Threshold method of Feature Selection')

#from sklearn.feature_selection import VarianceThreshold
#var_thres = VarianceThreshold(threshold=0.0)#to drop columns of 10% variance
#var_thres.fit(X_train)

#st.write('Number of features that does not have ZERO Variance')
#st.info(sum(var_thres.get_support()))

#st.write('Shape of our Training data')
#st.info(X_train.shape)
#st.write('Shape of our Test data')
#st.info(X_test.shape)

st.subheader('Fitting our best model : Random Forest Classifier')

from sklearn.ensemble import RandomForestClassifier

rd_clf = RandomForestClassifier(criterion = 'entropy', max_depth = 11, max_features = 'auto', min_samples_leaf = 2, min_samples_split = 3, n_estimators = 130)
rd_clf.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of random forest

rd_clf_acc = accuracy_score(y_test, rd_clf.predict(X_test))
st.write('Accuracy Score')
st.info(rd_clf_acc)

st.write('Classification Report')
st.info(classification_report(y_test, rd_clf.predict(X_test)))

st.write('Enter the the following Details')
#user_input = st.text_input("label goes here")
#st.header(user_input)
#inp = np.array([[user_input]])
#inp = inp.astype(float)







#pred = rd_clf.predict(inp)
#st.write(pred)
#X_test
st.write('Finished')
#df = {'You have Coronary disease' : 1,'You dont have Coronary disease' : 0}

#st.info(pred[df])