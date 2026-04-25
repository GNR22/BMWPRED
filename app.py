import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.figure_factory as ff
import time

# --- APP CONFIGURATION ---
st.set_page_config(page_title="BMW Sales BI Tool", layout="wide")
st.title("📊 Sales Performance Classification (BMW)")
st.markdown("Predictive modeling using KNN, SVM, and ANN algorithms.")

# --- 1. DATASET HANDLING ---
st.header("1. Dataset Handling & Sampling")
uploaded_file = st.file_uploader("Upload BMW_Car_Sales_Classification.csv", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    
    # --- CHRONOLOGICAL SORTING ---
    if 'Year' in df_raw.columns:
        df_raw = df_raw.sort_values(by='Year', ascending=True)
    
    # --- RECORD SLIDER ---
    total_rows = df_raw.shape[0]
    st.write("### 🎚️ Data Sampling Control")
    num_records = st.slider(
        "Select how many records to train on:", 
        min_value=500, 
        max_value=total_rows, 
        value=min(5000, total_rows), 
        step=500
    )
    
    df = df_raw.head(num_records)
    st.info(f"Using the first **{num_records}** chronological records out of {total_rows} available.")

    st.subheader("Dataset Preview (Sorted by Year)")
    st.dataframe(df, use_container_width=True)

    # --- 2. DATA PREPROCESSING ---
    st.divider()
    st.header("2. Data Preprocessing")
    
    target_col = 'Sales_Classification'
    
    if target_col not in df.columns:
        st.error(f"Critical Error: Column '{target_col}' not found in CSV!")
    else:
        cols_to_drop = [c for c in ['Model', 'Sales_Volume', 'Unnamed: 0', 'ID'] if c in df.columns]
        df_clean = df.drop(columns=cols_to_drop).dropna()
        
        st.write(f"**Target Variable:** `{target_col}`")
        st.write(f"**Excluded Features:** `{', '.join(cols_to_drop)}`")
        
        # --- THE TRAINING BUTTON ---
        if st.button("🚀 Train & Compare Models"):
            # ⏱️ START THE TIMER HERE
            start_time = time.time() 

            with st.spinner(f"Processing and training models on {num_records} records..."):
                
                # --- PREPARE DATA ---
                X = df_clean.drop(columns=[target_col])
                y = df_clean[target_col]
                
                if len(y.unique()) < 2:
                    st.error("❌ Error: The current sample only has one classification.")
                else:
                    # Encoding & Scaling
                    X = pd.get_dummies(X, drop_first=True)
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

                    # --- 3. MODEL IMPLEMENTATION ---
                    st.divider()
                    st.header("3. Predictive Model Implementation")
                    
                    models = {
                        "K-Nearest Neighbor (KNN)": KNeighborsClassifier(n_neighbors=5),
                        "Support Vector Machine (SVM)": SVC(kernel='rbf', probability=True, random_state=42),
                        "Artificial Neural Network (ANN)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
                    }
                    
                    results = []
                    tabs = st.tabs(list(models.keys()))
                    
                    for i, (name, model) in enumerate(models.items()):
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        acc = accuracy_score(y_test, y_pred)
                        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        cm = confusion_matrix(y_test, y_pred)
                        
                        results.append({
                            "Model": name, 
                            "Accuracy": acc, 
                            "Precision": prec, 
                            "Recall": rec, 
                            "F1-Score": f1
                        })
                        
                        with tabs[i]:
                            st.subheader(f"Detailed Metrics: {name}")
                            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                            col_m1.metric("Accuracy", f"{acc:.2%}")
                            col_m2.metric("Precision", f"{prec:.2%}")
                            col_m3.metric("Recall", f"{rec:.2%}")
                            col_m4.metric("F1-Score", f"{f1:.2%}")
                            
                            fig = ff.create_annotated_heatmap(
                                z=cm, 
                                x=[f"Predicted {c}" for c in le.classes_],
                                y=[f"Actual {c}" for c in le.classes_],
                                colorscale='Blues'
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    # ⏱️ STOP THE TIMER HERE (After all models are done)
                    end_time = time.time()
                    duration_seconds = end_time - start_time

                    # --- 4. MODEL COMPARISON & CONCLUSION ---
                    st.divider()
                    st.header("4. Model Comparison & Conclusion")
                    
                    # Display Time Result
                    if duration_seconds < 60:
                        st.info(f"⏱️ **Total Processing Time:** {duration_seconds:.2f} seconds")
                    else:
                        st.info(f"⏱️ **Total Processing Time:** {duration_seconds/60:.2f} minutes")

                    res_df = pd.DataFrame(results).set_index("Model")
                    col_table, col_chart = st.columns([1, 1])
                    
                    with col_table:
                        st.dataframe(res_df.style.highlight_max(axis=0, color='#2e7d32'), use_container_width=True)
                    
                    with col_chart:
                        st.bar_chart(res_df['Accuracy'])
                        
                    best_model = res_df['Accuracy'].idxmax()
                    best_acc = res_df['Accuracy'].max()
                    st.success(f"**Conclusion:** The **{best_model}** performed best with **{best_acc:.2%}** accuracy.")