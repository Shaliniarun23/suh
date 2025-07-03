# Streamlit dashboard code for BalanceBite Bar & Kitchen
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import seaborn as sns
import base64
import io

st.set_page_config(page_title='BalanceBite Dashboard', layout='wide')

@st.cache_data
def load_data():
    return pd.read_csv("balancebite_survey_synthetic_1000.csv")

df = load_data()

st.sidebar.title("Navigation")
tab = st.sidebar.radio("Select a module:", ["Data Visualisation", "Classification", "Clustering", "Association Rules", "Regression"])
