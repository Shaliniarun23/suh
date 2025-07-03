import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import os

# Set Streamlit page config
st.set_page_config(page_title="TasteMate Kitchen – End-to-End Analytics", layout="wide")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("synthetic_balancedbite_data.csv")
        return df
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return pd.DataFrame()

df = load_data()

# Check if data is loaded
if df.empty:
    st.stop()

# Navigation
tabs = st.tabs(["📊 Data Visualization", "⚙️ Classification", "🔍 Clustering", "🔗 Association Rules", "📈 Regression"])

# --- TAB 1: Data Visualization ---
with tabs[0]:
    st.header("📊 Descriptive Data Insights")
    
    st.subheader("Age Distribution by Gender")
    if 'Age' in df.columns and 'Gender' in df.columns:
        fig1 = px.histogram(df, x='Age', color='Gender', nbins=20, title="Age Distribution by Gender")
        st.plotly_chart(fig1)
    else:
        st.warning("Required columns 'Age' and 'Gender' are missing.")

# --- TAB 2: Classification ---
with tabs[1]:
    st.header("⚙️ Predict Loyalty Program Participation (Classification)")

    if 'Gender' in df.columns and 'Age' in df.columns and 'Loyalty_Program' in df.columns:
        # Prepare data
        X = pd.get_dummies(df[['Gender', 'Age']], drop_first=True)
        y = df['Loyalty_Program'].map({'Yes': 1, 'No': 0})
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        models = {
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier()
        }

        results = []
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            results.append({
                "Model": name,
                "Accuracy": round(accuracy_score(y_test, preds), 2),
                "Precision": round(precision_score(y_test, preds), 2),
                "Recall": round(recall_score(y_test, preds), 2),
                "F1 Score": round(f1_score(y_test, preds), 2)
            })

        st.dataframe(pd.DataFrame(results))
    else:
        st.warning("Required columns 'Gender', 'Age', and 'Loyalty_Program' are missing.")

# --- TAB 3: Clustering ---
with tabs[2]:
    st.header("🔍 Customer Segmentation via Clustering")
    st.write("Use KMeans clustering to identify distinct customer personas based on demographic and behavioral features.")

    cluster_cols = ['Age', 'Income', 'Order_Frequency', 'Avg_Spend_Per_Order', 'Weekly_Workout_Frequency']
    missing_cols = [col for col in cluster_cols if col not in df.columns]
    if missing_cols:
        st.warning(f"Missing columns for clustering: {', '.join(missing_cols)}")
    else:
        cluster_df = df[cluster_cols].dropna()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_df)
        kmeans = KMeans(n_clusters=3, random_state=42)
        cluster_df['Cluster'] = kmeans.fit_predict(scaled_data)
        fig2 = px.scatter(cluster_df, x='Income', y='Avg_Spend_Per_Order', color=cluster_df['Cluster'].astype(str),
                          title="Customer Clusters: Income vs Avg Spend")
        st.plotly_chart(fig2)

# --- TAB 4: Association Rules ---
with tabs[3]:
    st.header("🔗 Association Rule Mining (Coming Soon)")
    st.info("This feature is under development. You can display association rules here based on product combinations.")

# --- TAB 5: Regression ---
with tabs[4]:
    st.header("📈 Predict Spend Based on Income (Regression)")

    if 'Income' in df.columns and 'Avg_Spend_Per_Order' in df.columns:
        fig3 = px.scatter(df, x='Income', y='Avg_Spend_Per_Order',
                          trendline='ols', title="Average Spend per Order vs Income")
        st.plotly_chart(fig3)
    else:
        st.warning("Required columns 'Income' and 'Avg_Spend_Per_Order' are missing.")
