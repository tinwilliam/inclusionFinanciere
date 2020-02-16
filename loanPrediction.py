import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, f_classif

import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt

class LoanPrediction:

    def __init__(self, data):
        self.data = pd.read_csv(data)
        self.clean_data()
        self.split_train()

    def clean_data(self):
        self.data = self.data.dropna()
        self.data = self.data.drop('Loan_ID', axis=1)
        #print(self.data)
        self.gender = pd.get_dummies(self.data['Gender'])
        self.married = pd.get_dummies(self.data['Married'], prefix='Married')
        self.self_employed = pd.get_dummies(self.data['Self_Employed'], prefix='Self_employed')
        self.education = pd.get_dummies(self.data['Education'])
        self.property_area = pd.get_dummies(self.data['Property_Area'])
        self.data = pd.concat([self.data, self.gender, self.married, self.self_employed, self.education, self.property_area], axis=1)
        #self.data = pd.get_dummies(self.data)
        print(self.data)
        print(self.data.columns)
        self.data = self.data.drop(['Gender', 'Married', 'Self_Employed', 'Education', 'Property_Area', 'Dependents'], axis=1)

    def split_train(self):
        x = self.data.drop('Loan_Status', axis=1)
        y = self.data['Loan_Status']

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    def tree_decision(self):
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(self.x_train, self.y_train)
        print(clf.score(self.x_test, self.y_test))

    def random_forest(self):
        clf = RandomForestClassifier(n_estimators=100)
        clf = clf.fit(self.x_train, self.y_train)
        print(clf.score(self.x_test, self.y_test))

    def xgboost(self):
        model=xgb.XGBClassifier(random_state=42,learning_rate=0.01)
        model.fit(self.x_train, self.y_train)
        print(model.score(self.x_test,self.y_test))

    def select_k_best(self):
        selector = SelectKBest(chi2, k=7)
        selector.fit(self.x_train, self.y_train)
        x_new = selector.transform(self.x_train)
        self.x_train.columns[selector.get_support(indices=True)]

        # 1st way to get the list
        vector_names = list(self.x_train.columns[selector.get_support(indices=True)])
        print(len(self.x_train.columns))
        print(vector_names)

    def random_forest_f_classif(self):
        self.x_train = self.x_train[['CoapplicantIncome', 'LoanAmount', 'Credit_History', 'Married_No', 'Married_Yes', 'Rural', 'Semiurban']]
        self.x_test = self.x_test[['CoapplicantIncome', 'LoanAmount', 'Credit_History', 'Married_No', 'Married_Yes', 'Rural', 'Semiurban']]
        clf = RandomForestClassifier(n_estimators=100)
        clf = clf.fit(self.x_train, self.y_train)
        print(clf.score(self.x_test, self.y_test))

    def random_forest_chi2(self):
        self.x_train = self.x_train[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Credit_History', 'Married_No', 'Rural', 'Semiurban']]
        self.x_test = self.x_test[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Credit_History', 'Married_No', 'Rural', 'Semiurban']]
        clf = RandomForestClassifier(n_estimators=100)
        clf = clf.fit(self.x_train, self.y_train)
        print(clf.score(self.x_test, self.y_test))

    def xgboost_chi2(self):
        self.x_train = self.x_train[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Credit_History', 'Married_No', 'Rural', 'Semiurban']]
        self.x_test = self.x_test[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Credit_History', 'Married_No', 'Rural', 'Semiurban']]
        model=xgb.XGBClassifier(random_state=42,learning_rate=0.01)
        model.fit(self.x_train, self.y_train)
        print(model.score(self.x_test,self.y_test))

            

predictor = LoanPrediction('train_v3.csv')
# predictor.tree_decision()
# predictor.random_forest()
# predictor.xgboost()

predictor.select_k_best()

predictor.random_forest_chi2()
predictor.xgboost_chi2()