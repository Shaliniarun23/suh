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

import pandas as pd

@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_balancedbite_data.csv")  # Or your full filename
    return df



def load_data():
    df = pd.read_csv("synthetic_balancedbite_data.csv")
    return df

tabs = st.tabs(["📊 Data Visualization", "🤖 Classification", "🔍 Clustering", "🔗 Association Rules", "📈 Regression"])


# ----------------------- TAB 1: DATA VISUALIZATION --------------------------

with tabs[0]:
    st.header("📊 Descriptive Data Insights")

    # 1. Age Distribution by Gender
    st.subheader("Age Distribution by Gender")
    fig1 = px.histogram(df, x='Age', color='Gender', nbins=20, title="Age Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    # 2. Spend per Order by City
    st.subheader("Spend per Order by City")
    fig2 = px.box(df, x='City', y='Spend per Order', points="all", color='City', title="Spend Distribution Across Cities")
    st.plotly_chart(fig2, use_container_width=True)

    # 3. Cuisine Preferences Overview
    st.subheader("Cuisine Preferences")
    if 'Preferred Cuisines' in df.columns:
        cuisine_df = df['Preferred Cuisines'].str.get_dummies(sep=', ')
        cuisine_sum = cuisine_df.sum().sort_values(ascending=False)
        st.bar_chart(cuisine_sum)

    # 4. Order Frequency vs Satisfaction
    st.subheader("Order Frequency vs Satisfaction Score")
    if 'Order Frequency' in df.columns and 'Satisfaction (1-5)' in df.columns:
        fig3 = px.box(df, x='Order Frequency', y='Satisfaction (1-5)', color='Order Frequency', title="Satisfaction by Frequency")
        st.plotly_chart(fig3, use_container_width=True)

    # 5. Correlation Heatmap
    st.subheader("Correlation Matrix")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    corr_matrix = df[numeric_cols].corr()
    fig4 = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Among Numerical Features")
    st.plotly_chart(fig4, use_container_width=True)

    # 6. Gender vs Satisfaction
    st.subheader("Satisfaction Score by Gender")
    if 'Satisfaction (1-5)' in df.columns and 'Gender' in df.columns:
        fig5 = px.violin(df, x='Gender', y='Satisfaction (1-5)', box=True, points="all")
        st.plotly_chart(fig5, use_container_width=True)

    # 7. Weekly Workout vs Order Frequency
    st.subheader("Weekly Workout Frequency vs Order Frequency")
    if 'Weekly Workouts' in df.columns and 'Order Frequency' in df.columns:
        fig6 = px.scatter(df, x='Weekly Workouts', y='Order Frequency', color='City')
        st.plotly_chart(fig6, use_container_width=True)

    # 8. Drink Preferences
    st.subheader("Drink Type Preferences")
    if 'Drink Preference' in df.columns:
        fig7 = px.pie(df, names='Drink Preference', title="Preferred Drink Type")
        st.plotly_chart(fig7, use_container_width=True)

    # 9. Distribution of Fair Price Bundle
    st.subheader("Perceived Fair Price for Bundle")
    if 'Fair_Price_Bundle' in df.columns:
        fig8 = px.histogram(df, x='Fair_Price_Bundle', nbins=10, title="Fair Price Distribution")
        st.plotly_chart(fig8, use_container_width=True)

    # 10. Average Spend vs Income
    st.subheader("Average Spend per Order vs Income")
    if 'Monthly_Income' in df.columns and 'Spend per Order' in df.columns:
        fig9 = px.scatter(df, x='Monthly_Income', y='Spend per Order', trendline='ols', title="Spend vs Income")
        st.plotly_chart(fig9, use_container_width=True)


# ----------------------- TAB 2: CLASSIFICATION --------------------------

with tabs[1]:
    st.header("🤖 Customer Retention Classification")

    st.markdown("This model predicts whether a customer is likely to repeat an order based on their profile and behavior.")

    # Drop rows with missing target
    if 'Repeat Order' not in df.columns:
        st.warning("Target column 'Repeat Order' not found in dataset.")
    else:
        df_class = df.dropna(subset=['Repeat Order']).copy()

        # Encode target variable
        le = LabelEncoder()
        df_class['Repeat Order'] = le.fit_transform(df_class['Repeat Order'])  # Yes=1, No=0

        # Select predictors (exclude ID columns and target)
        features_to_drop = ['Repeat Order', 'Respondent_ID'] if 'Respondent_ID' in df.columns else ['Repeat Order']
        X = pd.get_dummies(df_class.drop(features_to_drop, axis=1), drop_first=True)
        y = df_class['Repeat Order']

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Define models
        models = {
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42)
        }

        # Train and evaluate models
        results = []
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            results.append({
                "Model": name,
                "Accuracy": round(report["accuracy"], 3),
                "Precision": round(report["weighted avg"]["precision"], 3),
                "Recall": round(report["weighted avg"]["recall"], 3),
                "F1-Score": round(report["weighted avg"]["f1-score"], 3)
            })

        st.subheader("🔍 Model Comparison Table")
        st.dataframe(pd.DataFrame(results))

        # Confusion Matrix
        model_choice = st.selectbox("Select model to view Confusion Matrix", list(models.keys()))
        selected_model = models[model_choice]
        selected_model.fit(X_train, y_train)
        y_pred_cm = selected_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_cm)

        st.subheader(f"📉 Confusion Matrix: {model_choice}")
        cm_df = pd.DataFrame(cm, index=["Actual: No", "Actual: Yes"], columns=["Predicted: No", "Predicted: Yes"])
        st.dataframe(cm_df)

        # ROC Curve Plot
        st.subheader("📈 ROC Curve for All Models")
        fig, ax = plt.subplots()
        for name, model in models.items():
            model.fit(X_train, y_train)
            if hasattr(model, "predict_proba"):
                y_scores = model.predict_proba(X_test)[:, 1]
            else:
                y_scores = model.decision_function(X_test)
            fpr, tpr, _ = roc_curve(y_test, y_scores)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)

        # Upload new data to predict Repeat Order
        st.subheader("📤 Upload New Data for Prediction")
        uploaded_file = st.file_uploader("Upload CSV (with same columns as training data except target)", type="csv")
        if uploaded_file:
            new_data = pd.read_csv(uploaded_file)
            new_data_encoded = pd.get_dummies(new_data, drop_first=True)
            missing_cols = set(X.columns) - set(new_data_encoded.columns)
            for col in missing_cols:
                new_data_encoded[col] = 0
            new_data_encoded = new_data_encoded[X.columns]
            new_data_scaled = scaler.transform(new_data_encoded)
            predictions = selected_model.predict(new_data_scaled)

            new_data['Predicted Repeat Order'] = le.inverse_transform(predictions)
            st.dataframe(new_data[['Predicted Repeat Order']])

            csv_out = new_data.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions", csv_out, "repeat_order_predictions.csv", "text/csv")


# ----------------------- TAB 3: CLUSTERING --------------------------

with tabs[2]:
    st.header("🔍 Customer Segmentation via Clustering")

    st.markdown("Use KMeans clustering to identify distinct customer personas based on demographic and behavioral features.")

    # Preprocess for clustering
    cluster_df = df.copy()
    cluster_cols = ['Age', 'Monthly_Income', 'Weekly Workouts', 'Order Frequency']
    cluster_df = cluster_df.dropna(subset=cluster_cols)

    if len(cluster_df) < 5:
        st.warning("Not enough data for clustering.")
    else:
        # Encode categorical variables
        cluster_data = pd.get_dummies(cluster_df[cluster_cols + ['City', 'Gender']], drop_first=True)

        # Scale
        scaler = StandardScaler()
        cluster_scaled = scaler.fit_transform(cluster_data)

        # Elbow method
        st.subheader("📉 Elbow Chart")
        distortions = []
        K = range(1, 11)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(cluster_scaled)
            distortions.append(kmeans.inertia_)

        fig_elbow = go.Figure(data=go.Scatter(x=list(K), y=distortions, mode='lines+markers'))
        fig_elbow.update_layout(title="Elbow Method for Optimal k", xaxis_title="Number of Clusters", yaxis_title="Inertia")
        st.plotly_chart(fig_elbow, use_container_width=True)

        # Choose number of clusters
        k_selected = st.slider("Select Number of Clusters", 2, 10, 3)
        kmeans_final = KMeans(n_clusters=k_selected, random_state=42)
        cluster_labels = kmeans_final.fit_predict(cluster_scaled)

        cluster_df['Cluster'] = cluster_labels

        st.subheader("🧠 Clustered Customer Data Preview")
        st.dataframe(cluster_df[['Age', 'Monthly_Income', 'Weekly Workouts', 'Order Frequency', 'City', 'Gender', 'Cluster']].head(10))

        # Cluster personas summary
        st.subheader("📌 Cluster Personas Summary")
        persona_summary = cluster_df.groupby('Cluster')[['Age', 'Monthly_Income', 'Weekly Workouts']].mean().round(1)
        persona_summary['Order Frequency Mode'] = cluster_df.groupby('Cluster')['Order Frequency'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'N/A')
        st.dataframe(persona_summary)

        # Download button
        st.subheader("📥 Download Clustered Dataset")
        csv_cluster = cluster_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV with Cluster Labels", csv_cluster, "clustered_customers.csv", "text/csv")


# ----------------------- TAB 4: ASSOCIATION RULES --------------------------

with tabs[3]:
    st.header("🔗 Association Rule Mining")

    st.markdown(
        "Discover hidden patterns in customer preferences by applying the Apriori algorithm on transaction-style data."
    )

    # Default columns that contain categorical preferences
    default_ar_cols = ['Preferred Cuisines', 'Order Influence', 'Delivery Issues']
    available_cols = [col for col in default_ar_cols if col in df.columns]

    if len(available_cols) < 2:
        st.warning("At least 2 valid columns required for association mining (e.g. Preferred Cuisines, Order Influence).")
    else:
        ar_cols = st.multiselect("Select Columns for Association Rules", available_cols, default=available_cols[:2])

        if len(ar_cols) >= 2:
            # Create transaction list
            transactions = df[ar_cols].fillna("").astype(str).apply(lambda x: ','.join(x), axis=1).str.split(',')
            transactions = transactions.apply(lambda x: list(set([i.strip() for i in x if i.strip()])))  # Clean duplicates

            # Transaction encoder
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            df_trans = pd.DataFrame(te_ary, columns=te.columns_)

            # Parameters
            min_supp = st.slider("Minimum Support", 0.01, 0.5, 0.1, 0.01)
            min_conf = st.slider("Minimum Confidence", 0.1, 1.0, 0.6, 0.05)

            # Run Apriori
            freq_items = apriori(df_trans, min_support=min_supp, use_colnames=True)
            rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)

            if rules.empty:
                st.warning("No rules found with current settings. Try lowering thresholds.")
            else:
                rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                rules_display = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
                rules_display = rules_display.sort_values(by='confidence', ascending=False)

                st.subheader("🔗 Top 10 Association Rules by Confidence")
                st.dataframe(rules_display.head(10))

                # Visualise lift distribution
                st.subheader("📈 Lift Distribution")
                fig_lift = px.histogram(rules, x='lift', nbins=20, title="Distribution of Lift Values")
                st.plotly_chart(fig_lift, use_container_width=True)
        else:
            st.info("Please select at least two columns to proceed with association rule mining.")


# ----------------------- TAB 5: REGRESSION --------------------------

with tabs[4]:
    st.header("📈 Spend Prediction via Regression Models")

    st.markdown("Estimate customer spending based on profile and behavior using various regression models.")

    if 'Spend per Order' not in df.columns:
        st.warning("Column 'Spend per Order' not found.")
    else:
        df_reg = df.copy()

        # Handle Spend per Order if categorical
        spend_map = {
            '<₹100': 50,
            '₹100–200': 150,
            '₹200–400': 300,
            '₹400–600': 500,
            '₹600+': 800
        }

        if df_reg['Spend per Order'].dtype == 'object':
            df_reg['Spend Num'] = df_reg['Spend per Order'].map(spend_map)
        else:
            df_reg['Spend Num'] = df_reg['Spend per Order']

        # Drop rows without Spend Num
        df_reg = df_reg.dropna(subset=['Spend Num'])

        # Prepare data
        y = df_reg['Spend Num']
        features_to_drop = ['Spend per Order', 'Spend Num', 'Repeat Order', 'Respondent_ID']
        features_to_drop = [col for col in features_to_drop if col in df_reg.columns]

        X = pd.get_dummies(df_reg.drop(features_to_drop, axis=1), drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        reg_models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=0.1),
            "Decision Tree Regressor": DecisionTreeRegressor(max_depth=5, random_state=42)
        }

        # Train and evaluate
        results = []
        for name, model in reg_models.items():
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            results.append({'Model': name, 'R² Score': round(score, 3)})

        st.subheader("📊 Model Performance")
        st.dataframe(pd.DataFrame(results).sort_values(by='R² Score', ascending=False))

        # Optional: Plot predictions of best model
        best_model_name = max(results, key=lambda x: x['R² Score'])['Model']
        best_model = reg_models[best_model_name]
        y_pred = best_model.predict(X_test)

        st.subheader(f"📈 Prediction Scatter Plot: {best_model_name}")
        fig_pred = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Spend', 'y': 'Predicted Spend'}, trendline='ols')
        fig_pred.update_layout(title="Actual vs Predicted Spend")
        st.plotly_chart(fig_pred, use_container_width=True)
