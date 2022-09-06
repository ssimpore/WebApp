import pickle

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
import streamlit as st


# get data
data = pd.read_csv("home_loan_data.csv")
df = data.copy()

X_ = df.drop(columns=['Loan_ID', 'Loan_Status'])
y_ = df[['Loan_Status']]

st.title("App Simple pour la prévision d'offre de crédit bancaire")

st.sidebar.header("Les parametres d'entrée du modèl")

def buildDataPreprocessing(arg: pd.DataFrame) -> pd.DataFrame:
    arg = arg.copy()

    cat_selector = make_column_selector(dtype_include=object)
    num_selector = make_column_selector(dtype_include=np.number)

    numeric_features = num_selector(arg)
    categorical_features = cat_selector(arg)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

    df_ = preprocessor.fit_transform(arg)

    cat_col_out = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features)

    df_ = pd.DataFrame(df_, columns=numeric_features + list(cat_col_out))

    return df_


def getUserInput(df_train):
    Gender = st.sidebar.selectbox('Genre', ('Male', 'Female'))
    Married = st.sidebar.selectbox('État civil', ('Yes', 'No'))
    Dependents = st.sidebar.selectbox('Personnes Dependantes', ('0', '1', '2', '3+'))
    Education = st.sidebar.selectbox('Éducation', ('Graduate', 'Not Graduate'))
    Self_Employed = st.sidebar.selectbox('Auto Entrepreneur', ('Yes', 'No'))
    ApplicantIncome = st.sidebar.slider('Révenus du demandant', min_value=150, max_value=81000, step=1000)
    CoapplicantIncome = st.sidebar.slider('Révenus du co-demandant', min_value=0, max_value=41000, step=1000)
    LoanAmount = st.sidebar.slider('Montant demandé', min_value=5, max_value=1000, step=10)
    Loan_Amount_Term = st.sidebar.slider('Durée du prêt', min_value=12, max_value=480, step=12)
    Credit_History = st.sidebar.selectbox('Historique de prêt', (1, 0))
    Property_Area = st.sidebar.selectbox('Localisation', ('Urban', 'Rural', 'Semiurban'))

    data = {'Gender': Gender,
            'Married': Married,
            'Dependents': Dependents,
            'Education': Education,
            'Self_Employed': Self_Employed,
            'ApplicantIncome': ApplicantIncome,
            'CoapplicantIncome': CoapplicantIncome,
            'LoanAmount': LoanAmount,
            'Loan_Amount_Term': Loan_Amount_Term,
            'Credit_History': Credit_History,
            'Property_Area': Property_Area}

    arg = pd.DataFrame(data, index=[0])
    arg = pd.concat([arg, df_train], axis=0)
    arg = buildDataPreprocessing(arg)
    arg = arg[:1]
    return arg


X_input = getUserInput(X_)

# Importation du model

model = pickle.load(open('prevision_credit.pkl', 'rb'))

prediction = model.predict(X_input)

prediction = int(prediction)

pred = {1: 'Prêt Accordée', 0: 'Prêt Refusé'}

# st.title("""Prevision:""")
prediction = pred[int(prediction)]

st.subheader("La *prevision* du credit est:")
st.write(prediction)
