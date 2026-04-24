# 📊 BMW Sales Performance Classification (BI Tool)

**Course:** IS 108 – INTELLIGENCE SYSTEM FINAL PROJECT SY 2025-2026
**Business Problem:** Sales Performance Classification

This repository contains a complete serverless Business Intelligence (BI) application that performs a predictive modeling process on BMW car sales data. It utilizes machine learning to predict whether a vehicle will have a "High" or "Low" sales performance based on its attributes.

---

## 🛠️ Tech Stack
This application is built using a modern, lightweight data science stack:
* **Language:** Python 3
* **Frontend/UI:** Streamlit (Serverless web application framework)
* **Data Manipulation:** Pandas & NumPy
* **Machine Learning:** Scikit-Learn (for KNN, SVM, and ANN)
* **Data Visualization:** Plotly (for interactive Confusion Matrix heatmaps)

---

## 📂 Dataset Source
The dataset used for this project is the BMW Car Sales Classification dataset. Due to file size limits, the CSV is not hosted directly in this repo.

Download Link: https://www.kaggle.com/datasets/sumedh1507/bmw-car-sales-dataset

File Name: BMW_Car_Sales_Classification.csv

# BMW Car Sales Prediction Dataset

## Column Descriptions

### 1. Car_Model
Specific model of the BMW car.

### 2. Year
Manufacturing year of the car.

### 3. Engine_Size
Engine capacity of the car in liters.

### 4. Fuel_Type
Type of fuel used by the car  
Examples: Petrol, Diesel, Hybrid.

### 5. Transmission
Transmission type of the car  
Examples: Manual, Automatic.

### 6. Mileage
Total distance driven by the car  
Measured in kilometers or miles.

### 7. Price
Listed price of the car in currency units.

### 8. Customer_Age_Group
Age group of the customer  
Examples: 18–25, 26–35, etc.

### 9. Income_Level
Income bracket of the customer  
Examples: Low, Medium, High.

### 10. Region
Geographic region or dealership location.

### 11. Previous_Owners
Number of previous owners of the car.

### 12. Car_Condition
Condition rating of the car  
Examples: Excellent, Good, Fair.

### 13. Sale_Outcome
Target variable of the dataset.

- **1** = Car Sold  
- **0** = Not Sold

## ⚙️ How the Application Works (The 4 Phases)

### 1. Dataset Handling & Sampling
Users upload the `BMW_Car_Sales_Classification.csv` file directly into the application. 
* **Chronological Sorting:** The dataset is immediately sorted by the `Year` column. In a real-world business scenario, forecasting is based on a timeline. By sorting chronologically, the data sample respects the historical timeline, ensuring that when the user selects a sample size (e.g., 2,000 records), the models are trained on a logical progression of older to newer models, rather than a randomized chaotic snapshot.
* **Dynamic Sampling:** A slider allows the user to scale the dataset from 500 to 50,000+ records to test how data volume affects model accuracy and training speed.

### 2. Data Preprocessing & The "XYZ" of Predicting
Before training, the data must be mathematically prepared. 

**Target Variable:** The application locks the target strictly to `Sales_Classification` to fulfill the specific rubric requirement for "Sales Performance Classification."

**Crucial Exclusions (Why we drop columns):**
* **Excluding `Sales_Volume` (Data Leakage):** The target variable ("High" or "Low" classification) is mathematically derived from the total sales volume. If we leave `Sales_Volume` in the dataset, the model simply reads the answer without learning anything. This is called *Data Leakage*. By removing it, we force the AI to genuinely *predict* the outcome using the car's features.
* **Excluding `Model` and `ID` (Overfitting):** If we include the car's name (e.g., "M5" or "X3"), the model will simply memorize that specific names historically perform well. By removing the name, the AI must learn the underlying patterns (e.g., *Expensive + Diesel + Low Mileage = High Sales*). This allows the application to accurately predict sales for brand-new car models it has never seen before.

**The "XYZ" of the Prediction Pipeline:**
* **X (The Features):** The independent variables (Price, Mileage, Engine Size, Region). The application converts text (like "Petrol") into numbers using `get_dummies` (One-Hot Encoding) and normalizes the numbers using `StandardScaler` so large numbers like Price don't overpower smaller numbers like Engine Size.
* **Y (The Target):** The dependent variable (`Sales_Classification`). We use `LabelEncoder` to turn "High" and "Low" into 1 and 0 so the algorithms can process the goal.
* **Z (The Prediction / $\hat{y}$):** After the model finds the mathematical relationship $f(X) = Y$, it outputs $Z$ (the prediction). We then compare $Z$ against the actual $Y$ in the testing set to grade the model.

### 3. Predictive Model Implementation
The dataset is split into 80% Training Data and 20% Testing Data. We simultaneously train three distinct algorithms to see which "learns" best:
1. **K-Nearest Neighbor (KNN):** A distance-based algorithm. It looks at a new car and finds the 5 closest cars in the dataset based on price, mileage, etc. It classifies the new car based on the majority classification of those neighbors.
2. **Support Vector Machine (SVM):** A geometric algorithm. It uses a Radial Basis Function (RBF) kernel to draw complex mathematical boundaries between "High" and "Low" performing cars in high-dimensional space. It is highly effective for medium-sized datasets.
3. **Artificial Neural Network (ANN):** A deep learning algorithm modeled after the human brain. We utilize a Multi-Layer Perceptron (MLP) with two hidden layers (64 neurons, then 32 neurons). It processes the data through these nodes to find highly complex, non-linear relationships (e.g., how Region combined with Fuel Type affects the outcome).

### 4. Model Comparison & Conclusion
The application evaluates the $Z$ predictions using strict business metrics:
* **Accuracy:** The total percentage of correct predictions.
* **Precision & Recall (Weighted):** Ensures the model isn't "cheating" by just guessing the most common class.
* **F1-Score:** The harmonic mean of precision and recall.
* **Confusion Matrix:** Plotted as an interactive heatmap. It shows exactly where models get "confused" (e.g., predicting a "High" sale when it was actually "Low"). 

The application concludes by displaying a bar chart comparison and dynamically printing a summary of the winning model based on the user's specific sample size.

---

## 🚀 Deployment & Local Setup

### Local Installation
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run app.py`

### Cloud Deployment
This application requires NO persistent database. It operates entirely in-memory (BYOD - Bring Your Own Data), making it fully compatible with serverless deployments like GitHub Pages, Vercel, or Streamlit Community Cloud.
