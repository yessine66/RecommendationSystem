import logging
from traceback import print_tb
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn import tree


def prepare_data(input_path):
    print("Loading and preparing data")
    df = pd.read_csv(input_path)
    df = df.replace(r'^\s*$', np.nan, regex=True)
    
    print("Encooding Data ")
    df['Sexe'] = pd.factorize(df['Sexe'])[0]
    df['Situation'] = pd.factorize(df['Situation'])[0]
    df['Ville'] = pd.factorize(df['Ville'])[0]
    df['Profession'] = pd.factorize(df['Profession'])[0]
    df['UtilisateurPotentiel'] = pd.factorize(df['UtilisateurPotentiel'])[0]
    df['TypeContrat'] = pd.factorize(df['TypeContrat'])[0]
    return df


def train_model(df, output_path):
    print("data splitting")
    X = df.drop(columns="TypeContrat")
    Y = df["TypeContrat"]

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1)

    print("Model training")
    model = DecisionTreeClassifier()
    model.fit(X_train,Y_train)
    
    print("Training results")
    predictions = model.predict(X_test)

    score = accuracy_score(Y_test,predictions)
    print(score)
    
    joblib.dump(model,"etafakna.joblib")
    print("model saved to ",output_path)


def load_model(model_path):
    return joblib.load(model_path)


def predict(userx, model):
    return model.predict([userx])


def run_training():
    logging.info("Training started...")
    df = prepare_data("../data/DataEtafakna.csv")
    train_model(df, "../models/etafakna.joblib")
    logging.info("Model finished training")
    
    
run_training()
    
    
