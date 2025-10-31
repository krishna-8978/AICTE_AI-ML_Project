# 🚗 Electric Vehicle (EV) Price Prediction Model

This project is the foundational machine learning task for a 4-week internship focused on EV data analysis and Generative AI applications. The core objective is to build a robust model that can accurately estimate the market price of an electric vehicle based purely on its performance specifications and manufacturer.

---

## 🎯 Problem Statement

The core challenge in the EV market is understanding the intrinsic value of technical specifications. This project addresses that by defining the problem as:

> **"Develop a machine learning regression model to accurately predict the Estimated US Market Price (`Estimated_US_Price`) of an Electric Vehicle (EV) based on its core technical specifications, including battery capacity, driving range, efficiency, and acceleration."**

---

## 💾 Data & Key Features

* **Dataset:** `EV_Cars_Data_Cleaned.csv`
* **Source:** Cleaned and processed data sourced from EV market statistics.
* **Target Variable ($\mathbf{y}$):** `Estimated_US_Price` (USD)
* **Most Influential Features ($\mathbf{X}$):**
    * `Top Speed` (Correlation: $0.77$)
    * `battery` (kWh) (Correlation: $0.66$)
    * `0 - 100` (Acceleration in seconds) (Correlation: $-0.57$)
    * `Range*` (km)
    * `title` (Manufacturer - handled via One-Hot Encoding)

---

## 🧠 Methodology and Model Performance (Week 1 Completion)

The project followed a Supervised Regression machine learning pipeline:

### 1. Exploratory Data Analysis (EDA)
* The price distribution was found to be **right-skewed**, indicating a need for potential log transformation in future refinement stages.
* A high positive correlation was established between **Top Speed**, **Battery Capacity**, and the final price.

### 2. Data Preprocessing
* **One-Hot Encoding** was applied to all relevant categorical features (`title` and `Drive_Configuration`) to convert them into a numerical format suitable for the model.
* The data was split into **80% Training** and **20% Testing** sets.

### 3. Model Training and Evaluation
A **Random Forest Regressor** was selected for its robust performance on complex, non-linear data.

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Model** | Random Forest Regressor | High-performance ensemble model used. |
| **R-squared ($R^2$) Score** | $0.6005$ | The model explains **$60.05\%$** of the variability in the EV price. |
| **Mean Absolute Error (MAE)** | **\$11,295.98** | On average, the model's prediction is off by approximately $\$11,300$. |

### Feature Importance Summary
The model confirmed that **Top Speed** is the single most important factor, contributing nearly **$50\%$** of the predictive power, followed by **battery** and **acceleration ($\mathbf{0 - 100}$)**.

---

## 🛠️ Next Steps (Week 2 Onwards)

The project will now move into refinement and application stages:

1.  **Model Refinement:** Implement **Hyperparameter Tuning** and data transformation (e.g., Log Transformation) to improve the $R^2$ score towards the target of $\mathbf{80\%}$ accuracy.
2.  **Generative AI Integration:** Develop a user-friendly frontend using **Streamlit** and integrate the prediction model, while setting up a **Generative AI Chatbot service** (using Gemini API or similar) to allow users to query EV data and ask general questions.

---

## 💻 Technologies Used

| Category | Tools/Libraries |
| :--- | :--- |
| **Programming** | Python (3.x) |
| **Data Handling** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn (RandomForestRegressor, train\_test\_split, metrics) |
| **Visualization** | Matplotlib, Seaborn |
| **Deployment (Planned)** | Streamlit, Gemini API |

### Required Dependencies

You can install all necessary dependencies using the following command after creating a `requirements.txt` file:

```bash
pip install -r requirements.txt
```
## 4. Environment Setup

### A. Environment Variables (Critical for Chatbot)

To run the Generative AI Chatbot section, you must set your API key as an environment variable.

1.  **Get Key:** Obtain a Gemini API Key from Google AI Studio.
2.  **Set Variable:** Set the key as an environment variable named `GEMINI_API_KEY`.

| Operating System | Command (Use in Terminal/PowerShell) |
| :--- | :--- |
| **macOS/Linux** | `export GEMINI_API_KEY="YOUR_KEY_HERE"` |
| **Windows (PowerShell)** | `$env:GEMINI_API_KEY="YOUR_KEY_HERE"` |

### B. Running the Application

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the Streamlit App:**
    ```bash
    python -m streamlit run app.py
    ``` requirements.txt
    ```
