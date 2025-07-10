# Global Superstore Data Analysis and Return Prediction

## 1. Introduction/Overview

This project focuses on performing a comprehensive Exploratory Data Analysis (EDA) and developing a predictive model for customer order returns using the "Global Superstore 2016" dataset. The primary goals are to uncover insights into sales patterns, customer behavior, and product performance, and to build a classification model that can predict whether an order is likely to be returned.

## 2. Dataset

*   **Source:** The data originates from the `global_superstore_2016.xlsx` Excel file.
    *   **Note:** This Excel file is an external dependency and is not included in this repository. It needs to be acquired separately to run the analysis.
*   **Sheets/Tables:** The analysis utilizes three main data sheets:
    *   **Orders:** Contains detailed information about each order, including sales, quantity, discount, profit, shipping details, customer segments, product categories, etc.
    *   **People:** Contains information about regional managers. (Note: While loaded, this table isn't extensively used in the provided analysis script beyond initial loading).
    *   **Returns:** Contains information about returned orders, specifically the `Order ID` of returned items.
*   **Relationships:**
    *   The `Orders` and `Returns` tables are linked via the `Order ID` column.
    *   An Entity Relationship Diagram (ERD) visualizing the data schema is available in `docs/ERD.drawio`.

## 3. Exploratory Data Analysis (EDA)

A detailed EDA was performed using the Python script `data_prep_report (1).py`. Key steps and findings include:

*   **Data Loading and Initial Inspection:** Loading data from Excel sheets, examining data types, and viewing initial rows.
*   **Missing Value Handling:** Identified and handled missing values. Notably, the 'Postal Code' column, having a high percentage of missing values, was dropped.
*   **Duplicate Removal:** Checked for and removed duplicate order IDs in the Orders and Returns data, as well as entire duplicate rows in the Orders data.
*   **Data Type Conversion:** Converted relevant columns (e.g., 'Order Priority', 'Ship Mode', 'Segment', 'Region', 'Category', 'Sub-Category') to categorical types for more efficient analysis.
*   **Data Merging:** Merged the `Orders` data with `Returns` data to create a comprehensive dataset where each order is flagged if it was returned.
*   **Descriptive Statistics:** Calculated and reviewed descriptive statistics for both numerical (Sales, Quantity, Discount, Profit, Shipping Cost) and categorical variables.
*   **Visualizations:**
    *   **Numerical Variables:** Histograms and box plots were used to understand the distribution and spread of sales, profit, shipping costs, etc.
    *   **Categorical Variables:** Count plots were used to visualize the frequency of different categories within ship modes, segments, regions, product categories, etc.
    *   **Relationships:**
        *   A correlation matrix heatmap was generated for numerical variables.
        *   A scatter plot with a regression line visualized the relationship between 'Sales' and 'Profit'.
        *   Bar plots showed return rates by customer 'Segment' and product 'Sub-Category'.
        *   Box plots illustrated 'Shipping Cost' distributions by 'Region' and 'Profit' distributions by 'Category'.
*   **Key Insights (Examples):**
    *   Identified the correlation between sales and profit.
    *   Analyzed orders with negative profit.
    *   Investigated return rates across different customer segments and product sub-categories.
    *   Explored shipping cost variations by region.
    *   Examined profit margins across product categories.

## 4. Predictive Modeling

A classification model was built to predict whether an order would be returned.

*   **Goal:** Predict if an order will be 'Returned' (Yes/No).
*   **Model:** A Random Forest Classifier was used.
*   **Preprocessing Steps:**
    *   **Target Encoding:** The 'Returned' column was mapped to numerical values (Yes: 1, No: 0).
    *   **Feature Selection:** A subset of relevant features was chosen, excluding high-cardinality or identifier variables. Selected features included 'Sales', 'Quantity', 'Discount', 'Shipping Cost', 'Ship Mode', 'Segment', 'Region', 'Category', 'Sub-Category', and 'Order Priority'.
    *   **Categorical Encoding:** One-hot encoding was applied to categorical features.
    *   **Numerical Scaling:** `StandardScaler` was used to scale numerical features.
    *   **Handling Class Imbalance:** The Synthetic Minority Over-sampling Technique (SMOTE) was employed to address the imbalance between returned and non-returned orders in the training data.
*   **Model Training & Evaluation:**
    *   The data was split into training (70%) and testing (30%) sets.
    *   The Random Forest model was trained on the resampled (SMOTE) training data.
    *   **Evaluation Metrics:**
        *   Classification Report (Precision, Recall, F1-score)
        *   Confusion Matrix
        *   ROC AUC Score and ROC Curve
*   **Performance & Feature Importance:**
    *   The model's performance was assessed using the metrics above, providing insights into its ability to correctly classify returns. (Specific scores can be found in the output of the script).
    *   Feature importance analysis was conducted to identify which factors most significantly influence the prediction of returns. Top features included 'Sales', 'Discount', and 'Shipping Cost'.

## 5. Tools and Libraries Used

*   **Python 3.x**
*   **Pandas:** For data manipulation and analysis.
*   **NumPy:** For numerical operations.
*   **Matplotlib:** For basic plotting.
*   **Seaborn:** For enhanced statistical visualizations.
*   **Scikit-learn (sklearn):** For machine learning tasks including:
    *   `train_test_split`
    *   `StandardScaler`
    *   `RandomForestClassifier`
    *   `classification_report`, `confusion_matrix`, `roc_auc_score`, `roc_curve`
*   **imblearn (imbalanced-learn):** For handling imbalanced datasets, specifically using `SMOTE`.
*   **Google Colab:** The analysis script (`data_prep_report (1).py`) contains elements suggesting it was developed in a Google Colab environment (e.g., `from google.colab import files`).

## 6. Skills Demonstrated/Learned

*   Data Cleaning and Preprocessing
*   Exploratory Data Analysis (EDA)
*   Data Visualization
*   Feature Engineering and Selection
*   Predictive Modeling (Binary Classification)
*   Model Evaluation and Interpretation
*   Handling Imbalanced Datasets (SMOTE)
*   Working with multi-sheet Excel data
*   Interpretation of business data to derive actionable insights.

## 7. How to Use/Reproduce

1.  **Obtain the Dataset:** Download the `global_superstore_2016.xlsx` file. This file is not included in the repository.
2.  **Environment Setup:**
    *   Ensure you have Python 3 installed.
    *   Install the necessary libraries:
        ```bash
        pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn openpyxl
        ```
        (Note: `openpyxl` is needed for `pd.read_excel`).
3.  **Run the Script:**
    *   Place the `global_superstore_2016.xlsx` file in the same directory as the `data_prep_report (1).py` script, or update the file path in the script.
    *   Execute the Python script:
        ```bash
        python "data_prep_report (1).py"
        ```
    *   The script will perform the EDA, train the model, and print evaluation results. It also includes code to save a cleaned CSV (`cleaned_global_superstore.csv`) and download it if run in a Colab environment.

## 8. Project Structure

```
.
├── README.md                   # This detailed overview of the project
├── data_prep_report (1).py     # Python script with EDA and predictive modeling
└── docs/
    └── ERD.drawio              # Entity Relationship Diagram for the dataset
```

## 9. Potential Future Work

*   **Hyperparameter Tuning:** Optimize the Random Forest model (or other models) using techniques like GridSearchCV or RandomizedSearchCV.
*   **Try Other Models:** Experiment with different classification algorithms (e.g., Logistic Regression, Gradient Boosting, XGBoost, Support Vector Machines).
*   **Advanced Feature Engineering:** Create new features that might improve model performance.
*   **Deeper Dive Analysis:** Conduct more specific analyses on high-return product categories, customer segments with high return rates, or reasons for returns if such data were available.
*   **Deployment:** If the model is sufficiently accurate, consider deploying it as a simple API or integrating it into a dashboard.
*   **Interactive Dashboard:** Create an interactive dashboard (e.g., using Dash/Plotly or Tableau) to visualize EDA findings and model predictions.
```
