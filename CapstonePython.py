from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns                 # Statistical visualisation
import matplotlib.pyplot as plt       # Visualisation
import numpy as np                    # Numerical operations
import pandas as pd                   # Data manipulation
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import streamlit as st
import pandas as pd
import joblib
import os

st.title("Vinstonia: Used Car Price Oracle")
st.write("This app predicts the price of used cars based on your inputs.")

# %% [markdown]
# # 🚗 Used Car Price Prediction
# ### Capstone Machine Learning Project | Data Science Bootcamp
#
# ---
#
# ## 📌 Project Overview
# This project builds a machine learning regression model to predict the price of used cars based on listing features provided by course instructure. The goal is to develop a model that is accurate, interpretable, and generalizable to unseen data.
#
# ## 🎯 Problem Statement
# Pricing a used car accurately is a common challenge for both buyers and sellers. Without a reliable reference, sellers may underprice their vehicles or buyers may overpay. By leveraging historical listing data and machine learning, this project aims to build a data-driven pricing tool that estimates fair market value based on a car's attributes — reducing guesswork and improving transparency in the second-hand car market.
#
# ## 📦 Dataset
# | Property | Detail |
# |----------|--------|
# | **Source** | `used_cars_dataset.csv` |
# | **Target Variable** | `Price` (continuous — regression problem) |
# | **Features** | Brand, Model, Year, Age, kmDriven, Transmission, Owner, FuelType, PostedDate, AdditionalInfo |
#
# ## 👤 Project Roles
# Since this is an individual project, all roles are handled by one person:
#
# | Role | Responsibility |
# |------|----------------|
# | Data Engineer | Data loading, cleaning & preprocessing |
# | Model Engineer | Model training, tuning & evaluation |
# | Analyst | EDA, feature engineering & interpretation |
# | Presenter / Documenter | Notebook narrative, visuals & conclusions |
#
# ## 📒 Notebook Structure
# 1. **Data Loading & Initial Exploration** — Load dataset, inspect shape, dtypes, and basic statistics
# 2. **Exploratory Data Analysis (EDA)** — Distributions, correlations, and visual insights
# 3. **Data Cleaning & Preprocessing** — Handle missing values, outliers, and encode categoricals
# 4. **Feature Engineering** — Create and select the most informative features
# 5. **Model Development** — Train and compare baseline and improved models
# 6. **Hyperparameter Tuning** — Optimize models using GridSearchCV / RandomizedSearchCV
# 7. **Evaluation & Error Analysis** — Interpret metrics, residuals, and overfitting/underfitting
# 8. **Insights & Conclusion** — Summarize findings and suggest next steps
#
# ---
# > 📅 **Submitted by:** *[Sibongile Ntsibande]*
# > 🗓️ **Date:** *[27 April 2026]*
# > 🐍 **Environment:** Python 3 | Jupyter Notebook | scikit-learn, pandas, numpy, matplotlib, seaborn

# %% [markdown]
# # 1. Data Loading & Initial Exploration
# ---
# > **Objective:** Load the raw dataset and audit its structure — checking data types, missing values, duplicates, and the initial distribution of the target variable (`Price`).

# %%
# ── Core Libraries ────────────────────────────────────────────────

# ── Scikit-learn ──────────────────────────────────────────────────


# Load the used car dataset
def load_data():
    data = pd.read_csv('used_cars_dataset.csv')
    return data


df_raw = load_data()
print(df_raw.head())

# %% [markdown]
# **Initial Observations:**
# The dataset contains null values across several columns and the `Price` column is stored as a string (e.g. `"₹ 2,50,000"`) — it will need to be stripped and converted to a numeric type before modelling.

# %%
# Check for data information to identify null values and data types

print(df_raw.info())


# %% [markdown]
# ### Missing Values Analysis
# Several columns contain a significant number of missing values. High null counts can bias results or cause errors in the scikit-learn pipeline.
#
# **Strategy:** For each affected column, I will assess the percentage of missing data and its distribution — then decide whether to drop, impute with mean/median, or impute with mode accordingly.
#
#
#
#
#
#
#
#

# %%
# Check exactly how many missing values there are in each column
print(df_raw.isnull().sum())

# %% [markdown]
# ### Column Review & Drop Strategy
# A few columns stand out immediately:
# - **AdditionalInfo** — contains dense, unstructured mixed-type text. I will attempt to extract useful features from it before dropping it.
# - **Year & Age** — are repetitive. I will keep `Age` as it requires no transformation and drop `Year`.
#
# For all remaining columns I will: handle nulls, assess outliers, and plot each feature against `AskPrice` during EDA before making final drop decisions.
#
#
#
#
#
#
#
#
#
#

# %%
# Converting columns to the correct data types


def convert_to_appropriate_dtype(dfraw):

    df_convert = dfraw.copy()


# Convert 'kmDriven' to numeric, removing commas and 'km' before conversion
    df_convert['kmDriven'] = df_convert['kmDriven'].astype(
        str).str.replace(',', '', regex=False)
    df_convert['kmDriven'] = df_convert['kmDriven'].str.replace(
        'km', '', regex=False)
    df_convert['kmDriven'] = pd.to_numeric(
        df_convert['kmDriven'], errors='coerce')

# Convert 'PostedDate' to datetime, coercing errors to NaT
    df_convert['PostedDate'] = pd.to_datetime(
        df_convert['PostedDate'], format='%b-%y', errors='coerce')

# Convert 'AskPrice' to numeric, removing commas and '₹' before conversion
    df_convert['AskPrice'] = df_convert['AskPrice'].astype(
        str).str.replace(',', '', regex=False)
    df_convert['AskPrice'] = df_convert['AskPrice'].str.replace(
        '₹', '', regex=False)
    df_convert['AskPrice'] = pd.to_numeric(
        df_convert['AskPrice'], errors='coerce')

    return df_convert


df_correct_dtypes = convert_to_appropriate_dtype(df_raw)
print(df_correct_dtypes.info())


# %% [markdown]
# The `kmDriven` and `AskPrice` columns were converted to numeric and `PostedDate` was converted to datetime, putting all columns into formats suitable for statistical analysis and model training.
#
# Errors were coerced to `NaN`/`NaT` to prevent the pipeline from crashing on unexpected strings, allowing us to handle these values in the next cleaning phase.

# %% [markdown]
# ### Missing Values Strategy
# For each column, I will calculate the percentage of missing values and handle them as follows:
#
# | Missing % | Action | Reasoning |
# |-----------|--------|-----------|
# | < 5% | Drop rows | Too small to meaningfully impact the model |
# | 5% – 40% | Impute (mean/median/mode) | Too valuable to discard — enough data exists to estimate reliably |
# | > 40% | Drop column | More noise than signal — unless `AdditionalInfo` can be used to recover values |
#
#
#
#
#
#

# %%
# checking the percentage of missing values in each column to decide on a strategy for handling them

print(df_correct_dtypes.isnull().sum() / len(df_correct_dtypes) * 100)


# %% [markdown]
# The `Model` and `Owner` columns have a high percentage of missing values but are potentially important predictors of price. Rather than dropping them immediately, I will impute missing entries with a new category `'unknown'` and re-evaluate their usefulness after EDA.
#
# *Note: This deviates from the >40% drop rule above — the decision will be revisited after plotting these features against `AskPrice`.*
#
#

# %%
# Summary statistics for all numeric columns
df_correct_dtypes.describe()

# %% [markdown]
# **Numeric Columns:**
# - **Age:** Mean and median are close, suggesting a roughly symmetrical distribution. However, the maximum value of 124 years is unrealistic and will skew the mean — I will use the **median** for imputation to be safe. Outliers will be addressed in the cleaning phase.
# - **kmDriven:** A notable gap between mean and median indicates the presence of outliers pulling the mean up. I will use the **median** for imputation.
# - **AskPrice:** A large difference between mean and median suggests extreme high-end prices are skewing the distribution. No imputation needed but outliers must be handled.
#
# **Categorical Columns:**
# All categorical columns have missing values above 20%. Imputing with mode risks introducing bias, so I will assign a new category `'unknown'` to preserve the distribution.
#
# **PostedDate:**
# Will be imputed with the **mode** (most frequent date) — a timestamp is required for feature extraction so `'unknown'` is not a viable option here.
#
# **Year:**
# Left as-is — this column will be dropped later as it is redundant with `Age`.
#
#
#
#
#
#
#
#

# %%
# Filling missing values across all columns


def filling_columns(df_yes_dytpes):
    df_fc = df_yes_dytpes.copy()


# filling categorical columns with 'Unknown'
    cat_cols = ['Brand', 'model', 'Transmission', 'Owner', 'FuelType']
    for col in cat_cols:
        df_fc[col] = df_fc[col].fillna('Unknown')

# filling numerical columns with the median value
    df_fc['Age'] = df_fc['Age'].fillna(df_fc['Age'].median())
    df_fc['kmDriven'] = df_fc['kmDriven'].fillna(df_fc['kmDriven'].median())

# filling the date time column
    df_fc['PostedDate'] = df_fc['PostedDate'].fillna(
        df_fc['PostedDate'].mode()[0])

    return df_fc


print("Missing values after fill:")
df_filled = filling_columns(df_correct_dtypes)
print(df_filled.isnull().sum())

# %%
# Checking if all of the dates are in the same year
print(df_filled['PostedDate'].value_counts())

# %% [markdown]
# Most entries are from 2024, so the year adds no value. I will extract the month from `PostedDate` and drop the full date column — simplifying the data without losing useful information.
#
#

# %%


def extract_month(df_f):
    df_inside = df_f.copy()

    df_inside['posted_month'] = df_inside['PostedDate'].dt.month
    df_inside = df_inside.drop(['PostedDate'], axis=1)
    return df_inside


df_filled2 = extract_month(df_filled)
print(df_filled2.head())


# %% [markdown]
# ### 2. Exploratory Data Analysis (EDA)
# ---
# > **Objective:** Understand the distribution of individual features, identify relationships with `AskPrice`, and inform final feature selection.
#
# This section covers univariate analysis to detect and handle outliers, multivariate analysis to assess each feature's relationship with `AskPrice`, a correlation heatmap for numeric features, and a final feature selection decision.
#
#

# %%
# Dealing with the numerical columns

def plot_numerical_distributions(df2):

    df_pltd = df2.copy()
    num_cols = ['Age', 'kmDriven', 'AskPrice']

    for col in num_cols:
        plt.figure(figsize=(12, 4))

        # Histogram
        plt.subplot(1, 2, 1)
        # kde=True adds a density curve
        sns.histplot(df_pltd[col], kde=True, color='skyblue')
        plt.title(f'{col} Distribution')

        # Boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df_pltd[col], color='lightcoral')
        plt.title(f'{col} Outliers')
        plt.tight_layout()

        plt.show()


plot_numerical_distributions(df_filled2)

# %%
# Statistical summary of Age to inform outlier thresholds
print(df_filled2['Age'].describe())

# %% [markdown]
# Research shows that used cars are most commonly sold at 3–5 years, with a secondary trend of older vehicles (8–9+ years) remaining popular — particularly classic and budget models common in the market (e.g. Tazz, Uno).
#
# To account for this, I will use an IQR multiplier of **3 instead of 1.5** when removing outliers, preserving legitimate older car listings while still eliminating unrealistic values like the 124-year maximum observed earlier.
#
#

# %%
# removing outliers using the IQR method for the 'Age' column


def remove_outliers_iqr(df_iqr):
    df_iqr_copy = df_iqr.copy()

    Q1 = df_iqr_copy['Age'].quantile(0.25)
    Q3 = df_iqr_copy['Age'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    return df_iqr_copy[(df_iqr_copy['Age'] >= lower_bound) & (df_iqr_copy['Age'] <= upper_bound)]


df_filled3 = remove_outliers_iqr(df_filled2)
print(df_filled3['Age'].describe())


# %% [markdown]
# The `kmDriven` distribution mirrors `Age` — a bell curve peaking between 0 and 200,000 km, with the boxplot revealing a maximum near 980,000 km. Given that 75% of values fall below 79,000 km, a near-million kilometre reading is unrealistic and likely an error.
#
# Research supports a benchmark of 15,000–20,000 km per year, meaning a typical 5-year-old car would have 75,000–100,000 km — consistent with what we see in the `Age` column.
#
# A standard IQR multiplier of 1.5 would cap at roughly 120,000 km, which is too aggressive and would remove a large portion of valid listings. I will instead apply a hard cap of **200,000 km** as the upper bound, which is more realistic and preserves more usable data, it also reflects real market transactions for used cars.

# %%
# Capping kmDriven at 200,000 km
df_filled3 = df_filled3[df_filled3['kmDriven'] <= 200000]
print(df_filled3['kmDriven'].describe())

# %% [markdown]
# `AskPrice` is our target variable so minimising data loss here is a priority. The distribution is heavily right-skewed — the gap between mean and median suggests a significant number of high-end luxury vehicles are pulling the mean up.
#
# Applying the standard IQR method directly would remove legitimate high-value listings, leaving the model unable to predict prices for premium brands.
#
# To address this, I will apply a **log transformation** to normalise the distribution before applying IQR-based outlier removal — preserving as much pricing data as possible across the full market range.

# %%
# Log-transform AskPrice to reduce skewness
# log1p is used to safely handle any zero values
df_filled3['Log_AskPrice'] = np.log1p(df_filled3['AskPrice'])

# Plot original vs log-transformed distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df_filled3['AskPrice'], kde=True).set_title(
    "Original AskPrice (Skewed)")

plt.subplot(1, 2, 2)
sns.histplot(df_filled3['Log_AskPrice'], kde=True).set_title(
    "Log AskPrice (Normalised)")

plt.tight_layout()
plt.show()

# Remove outliers from log-transformed AskPrice using standard IQR method
Q1_log = df_filled3['Log_AskPrice'].quantile(0.25)
Q3_log = df_filled3['Log_AskPrice'].quantile(0.75)
IQR_log = Q3_log - Q1_log

lower_log = Q1_log - 1.5 * IQR_log
upper_log = Q3_log + 1.5 * IQR_log

df_filled3 = df_filled3[(df_filled3['Log_AskPrice'] >= lower_log) &
                        (df_filled3['Log_AskPrice'] <= upper_log)]

print(df_filled3['Log_AskPrice'].describe())


# %%
# Strip whitespace and standardise casing for categorical columns
cat_cols = ['Brand', 'model', 'Transmission', 'Owner', 'FuelType']
for col in cat_cols:
    df_filled3[col] = df_filled3[col].str.strip().str.title()

# Drop redundant columns and save as cleaned dataset
df_cleaned = df_filled3.copy()
df_cleaned = df_cleaned.drop(['Year', 'AdditionInfo'], axis=1)
print(df_cleaned.columns)

# %% [markdown]
# ### Univariate Analysis — Categorical Columns
# The data is now cleaned. The following plots show the distribution of each categorical feature to identify dominant categories and any remaining imbalances.

# %%
# plotting the catigorical columns
cat_cols = ['Brand', 'model', 'Transmission',
            'Owner', 'FuelType', 'posted_month']


for col in cat_cols:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df_cleaned, x=col,
                  order=df_cleaned[col].value_counts().index)
    plt.title(f'{col} Distribution')
    plt.xticks(rotation=90)
    plt.show()

# displaying the actual counts of each category in the categorical columns
for col in cat_cols:
    print(f"{col} value counts:")
    print(df_cleaned[col].value_counts())
    print("\n")

# %%
# Group rare and luxury brands to reduce cardinality in the Brand column
brand_counts = df_cleaned['Brand'].value_counts()

# Brands with fewer than 20 listings are considered rare
rare_brands = brand_counts[brand_counts <
                           20].index

# Ultra-luxury brands are separated to preserve their pricing signal
luxury_list = ['Rolls-Royce', 'Lamborghini', 'Bentley', 'Maserati', 'Ferrari']


def group_brands(brand):
    if brand in luxury_list:
        return 'Ultra_Luxury'
    if brand in rare_brands:
        return 'Other_Rare'
    return brand


df_cleaned['Brand_Grouped'] = df_cleaned['Brand'].apply(group_brands)
print(df_cleaned['Brand_Grouped'].value_counts())

# %% [markdown]
# The `Brand` column contains several manufacturers with very few listings, which can introduce noise and cause the model to overfit on rare data points. To address this, brands were grouped as follows:
#
# - **Ultra_Luxury** — High-end brands (e.g. Ferrari, Rolls-Royce) are grouped together to preserve their unique pricing signal without inflating the feature space.
# - **Other_Rare** — Any brand with fewer than 20 listings is consolidated into a single category.
#
# This reduces cardinality while retaining the most statistically significant brand information.

# %%
# Seeing how many models have fewer than 5 listings to determine if this column is useful or just noise
model_counts = df_cleaned['model'].value_counts()
rare_models_count = len(model_counts[model_counts < 5])
total_models = len(model_counts)

print(f"Total Unique Models: {total_models}")
print(f"Models with fewer than 5 listings: {rare_models_count}")
print(
    f"Percentage of 'Rare' models: {(rare_models_count / total_models) * 100:.2f}%")

# %% [markdown]
# ### Univariate Analysis — Key Observations
#
# **Shared concern — High Unknown counts (`Transmission`, `Owner`, `FuelType`, `Brand`):**
# High volumes of `Unknown` entries are problematic for two reasons — simpler models may struggle to learn meaningful patterns from them, and they risk masking the true relationship each feature has with price. A more complex algorithm could potentially handle this, but the decision will be made after evaluating correlation with `Log_AskPrice`.
#
# **Shared concern — High Cardinality (`Brand`, `Model`):**
# High cardinality leads to an explosion of dummy variables during one-hot encoding, many of which will have very few listings beneath them — adding noise rather than signal. Despite this, both columns are potentially important price predictors so final drop decisions will be made after multivariate analysis.
#
# **Model** carries the most risk — long-tail analysis shows the majority of unique models have fewer than 5 listings, making it difficult for any model to learn reliably from this column.

# %% [markdown]
# ### Multivariate analysis
#
#

# %%
# Multivariate analysis — numerical features vs Log_AskPrice
# regplot overlays a regression line to visualise the direction and strength of each relationship


# 1. Price vs. kmDriven
plt.figure(figsize=(10, 6))
sns.regplot(data=df_cleaned, x='kmDriven', y='Log_AskPrice',
            scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
plt.title('Relationship: kmDriven vs LogPrice')
plt.show()

# 2. Price vs. Age
plt.figure(figsize=(10, 6))
sns.regplot(data=df_cleaned, x='Age', y='Log_AskPrice',
            scatter_kws={'alpha': 0.3}, line_kws={'color': 'green'})
plt.title('Relationship: Age vs LogPrice')
plt.tight_layout()
plt.show()

# %%
# Boxplots showing the relationship between each categorical feature and Log_AskPrice
# Sorted by median price to make trends easier to interpret

cat_to_plot = ['Brand_Grouped', 'Transmission',
               'FuelType', 'Owner', 'posted_month']

for col in cat_to_plot:
    plt.figure(figsize=(12, 6))
    # Sorting by median price makes the trends much easier to see
    order = df_cleaned.groupby(
        col)['Log_AskPrice'].median().sort_values().index

    sns.boxplot(data=df_cleaned, x=col, y='Log_AskPrice', order=order)
    plt.title(f'{col} vs Log_AskPrice (Sorted by Median)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# %%
# Correlation heatmap for numerical features
numerical_cols = ['Log_AskPrice', 'kmDriven', 'Age']
plt.figure(figsize=(8, 6))
sns.heatmap(df_cleaned[numerical_cols].corr(),
            annot=True, cmap='RdYlGn', fmt=".2f")
plt.title('Correlation Heatmap — Numerical Features vs Log_AskPrice')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Multivariate Analysis & Final Feature Selection
#
# **kmDriven:** Negative correlation with `Log_AskPrice` — as mileage increases, price decreases. Wide scatter suggests other factors also influence price, and a slight fan shape indicates some heteroscedasticity. ✅ **Retained**
#
# **Age:** Also negatively correlated — older cars are priced lower. Steeper regression line than `kmDriven` suggests it may prove to be the stronger predictor. ✅ **Retained**
#
# **Brand:** High variance across brand tiers confirms a strong relationship with price. Ultra-luxury brands sit significantly higher, while `Unknown` clusters in the lower-to-mid range. Replaced by `Brand_Grouped` to preserve luxury signal without high cardinality. ✅ **Retained (as Brand_Grouped)**
#
# **Transmission:** Automatic is priced higher than manual, with `Unknown` sitting between the two — confirming it carries its own pricing signal and should be retained as a separate category. ✅ **Retained**
#
# **FuelType:** Clear price variation across fuel types. `Hybrid` and `Hybrid/CNG` were merged — same engine type, splitting them unnecessarily dilutes the signal. ✅ **Retained**
#
# **Owner:** Clear downward price trend as ownership increases. `Unknown` median sits close to first owner, suggesting these are likely newer cars. Retained as its own category. ✅ **Retained**
#
# **posted_month:** May shows a notably higher price distribution despite lower volume, suggesting seasonal pricing patterns worth capturing. ✅ **Retained**
#
# **model:** Too high-cardinality for one-hot encoding — target encoded instead to capture pricing signal without creating hundreds of dummy variables. ✅ **Retained (target encoded)**
#
# | Dropped Column | Reason |
# |----------------|--------|
# | `Brand` | Replaced by `Brand_Grouped` |
# | `Year` | Redundant with `Age` |
# | `AdditionInfo` | Unstructured text — extraction too complex, noted as future improvement |
# | `AskPrice` | Raw target — `Log_AskPrice` used instead |

# %% [markdown]
# ### Encoding & Train/Test Split
# Data is split before encoding to prevent data leakage. Target encoding for `model` and one-hot encoding for remaining categoricals is then applied to the training set and mapped to the test set.

# %%
# Merge Hybrid/CNG into Hybrid — same engine type, splitting reduces signal
df_cleaned['FuelType'] = df_cleaned['FuelType'].replace('Hybrid/CNG', 'Hybrid')

# Drop original Brand column — replaced by Brand_Grouped during feature engineering
df_final = df_cleaned.drop(columns=['Brand'])
print(df_final.columns)

# %% [markdown]
# ### Splitting & Encoding the Data
#
# The feature matrix (`X`) and target variable (`y = Log_AskPrice`) are defined and split into training and test sets before any encoding is applied — this is critical to prevent **data leakage**.
#
# - **`model` → Target Encoding:** Each model is replaced with its mean `Log_AskPrice` calculated from training data only, then mapped to the test set. This captures the column's pricing signal without creating hundreds of dummy variables.
# - **Remaining categoricals → One-Hot Encoding:** `Brand_Grouped`, `Transmission`, `FuelType` etc. have manageable cardinality and are one-hot encoded safely.
#
# Unseen models in the test set are filled with the global training mean as a fallback.
#
#

# %%
# 1. First, we split the data (Before any encoding)
X_raw = df_final.drop(['Log_AskPrice', 'AskPrice'], axis=1)
y_raw = df_final['Log_AskPrice']

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)

# Encode model column
# Calculate the average LogPrice for each model using ONLY training data
model_means = y_train.groupby(X_train['model']).mean()

# Map those averages back to the 'model' column
X_train['model_encoded'] = X_train['model'].map(model_means)
X_test['model_encoded'] = X_test['model'].map(model_means)

# If a model in the Test set wasn't in the Train set, fill it with the global average
X_train['model_encoded'] = X_train['model_encoded'].fillna(y_train.mean())
X_test['model_encoded'] = X_test['model_encoded'].fillna(y_train.mean())

# 3. Drop the original text 'model' column
X_train = X_train.drop('model', axis=1)
X_test = X_test.drop('model', axis=1)

# 4. One-Hot Encode the remaining categories
# This handles Brand_Grouped, Transmission, FuelType, etc.
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# 5. Final Alignment
# This ensures X_test has the exact same columns as X_train
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")

# %%


# Initialize and train
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
y_pred = lr_model.predict(X_test)

# Evaluate (Converting back from Log to actual Price for readability)
mae_lr = mean_absolute_error(np.exp(y_test), np.exp(y_pred))
r2_lr = r2_score(y_test, y_pred)

print(f"R-squared Score: {r2:.4f}")
print(f"Average Error (MAE): {mae:.2f}")


# %% [markdown]
#  The model explains **67% of the pattern that predicts AskPrice** based on the features provided — a moderate result. The remaining 33% represents pricing patterns the model could not capture, likely due to the non-linear relationships we identified in our correlation analysis.
#
# The MAE of ₹318,690 is relatively high, which is consistent with the missing 33% — the model is struggling most with cars whose prices are driven by complex interactions between features like brand, age and transmission that a straight line cannot fully represent. A Decision Tree should improve both metrics by capturing these non-linear patterns.

# %%
# Compare train vs test R² to check for overfitting or underfitting

train_score = lr_model.score(X_train, y_train)
test_score = lr_model.score(X_test, y_test)

print(f"Train R-squared: {train_score:.4f}")
print(f"Test R-squared: {test_score:.4f}")

# %% [markdown]
# The train and test R² scores are nearly identical (0.673 vs 0.669), confirming the model is neither overfitting nor underfitting — it generalises well but has reached the ceiling of what a linear model can capture with this data. Cross-validation will confirm this stability, after which we will move to a Decision Tree to improve performance by capturing the non-linear patterns a straight line cannot represent.

# %% [markdown]
#

# %%
# Cross validation to check for stabil

# We use X_train and y_train from your encoding cell
# cv=5 means it will split the training data into 5 pieces and rotate
cv_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='r2')


print(f"Individual Fold R2 Scores: {cv_scores}")
print(f"---")
print(f"Mean CV R2 Score: {round(cv_scores.mean(), 4)}")
print(f"Standard Deviation: {round(cv_scores.std(), 4)}")

# %% [markdown]
# Cross-validation trains 5 independent models on different slices of the training data, each validated on a different fold. The consistently similar R² scores across all 5 folds confirm that the Linear Regression result is stable and not a product of a lucky train/test split.
#
# The low standard deviation and mean CV R² close to 0.667 suggest the model has reached its ceiling — it does not have the capacity to capture the complex, non-linear relationships between features and `AskPrice`.
#
# A Decision Tree will be better suited here for three reasons:
# - It handles non-linear relationships naturally by splitting data into groups rather than fitting a straight line
# - It is less sensitive to unscaled features and missing value categories like `Unknown`
# - It can capture interactions between features (e.g. brand + age combined) that Linear Regression treats independently
#
#

# %% [markdown]
# ### Improved Model — Decision Tree Regressor
#
# A Decision Tree is chosen as the improved model for the following reasons:
#
# - Handles **non-linear and complex relationships** between features and price naturally
# - More robust to **unscaled features** and categorical noise (e.g. `Unknown` entries) than Linear Regression
# - Can capture **interactions between features** (e.g. brand + age combined) that Linear Regression treats independently
# - Provides **feature importance scores**, allowing us to identify which attributes most strongly influence price — useful for improving data collection in future
# - Expected to improve on the Linear Regression R² of 0.667 given the complexity of relationships in this dataset

# %%


# Train the model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Get predictions from the trained model
y_pred_dt = dt_model.predict(X_test)

# Get the R-squared and MAE for the Decision Tree model
r2_dt = r2_score(y_test, y_pred_dt)
mae_dt = mean_absolute_error(np.exp(y_test), np.exp(y_pred_dt))

print(f"Decision Tree R-squared: {r2_dt:.4f}")
print(f"Decision Tree MAE: {mae_dt:,.2f}")

# %%

# Checking for over or underfitting by comparing train and test scores

train_score_dt = dt_model.score(X_train, y_train)
test_score_dt = dt_model.score(X_test, y_test)

print(f"Train R-squared: {train_score_dt:.4f}")
print(f"Test R-squared: {test_score_dt:.4f}")

# %% [markdown]
# #### Decision Tree — Results
#
# | Metric | Linear Regression | Decision Tree |
# |--------|------------------|---------------|
# | R² | 0.667 | 0.853 |
# | MAE | ₹318,690 | ₹115,814 |
#
# The Decision Tree explains **85% of the pattern that determines AskPrice** given the provided features — a significant improvement over the Linear Regression baseline. The MAE has also dropped dramatically, meaning predictions are considerably closer to actual prices on average.
#
# However, the gap between train R² (0.977) and test R² (0.853) is a clear sign of **overfitting** — the model has memorised patterns in the training data that don't fully generalise to unseen cars. This is common in unpruned Decision Trees as they grow branches until every training sample is perfectly classified.
#
# Hyperparameter tuning will be used to prune the tree — limiting its depth and complexity to reduce overfitting while maintaining as much predictive power as possible.
#

# %%
# Actual vs predicted prices — points close to the black line indicate accurate predictions

actual = np.exp(y_test)
predicted = np.exp(y_pred_dt)

plt.figure(figsize=(8, 6))
plt.scatter(actual, predicted, alpha=0.7, color='pink')

plt.plot(
    [actual.min(), actual.max()],
    [actual.min(), actual.max()], color='black'

)

plt.xlabel("Actual Price (INR)")
plt.ylabel("Predicted Price")
plt.title("Decision Tree: Actual vs Predicted Prices")
plt.tight_layout()
plt.show()

# %% [markdown]
# The scatter plot confirms the MAE result — the majority of predictions cluster close to the perfect fit line, particularly in the lower price range where the model performs most reliably.
#
# Predictions become more scattered at higher price points, which is consistent with our earlier observation that luxury and high-end vehicles introduce complexity that the model struggles to capture. This reinforces the case for hyperparameter tuning to improve performance across the full price range.

# %% [markdown]
# ### Hyperparameter Tuning

# %%

# Define the parameters we want to test
param_grid = {
    'max_depth': [None, 10, 20, 30],  # limits the number of levels in the tree
    # the node will only be split if it has at least this many samples
    'min_samples_split': [2, 5, 10],
    # the node will only be a leaf if it has at least this many samples
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}


# Run GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(DecisionTreeRegressor(random_state=42),
                           param_grid,
                           cv=5,
                           scoring='r2',
                           n_jobs=-1)


grid_search.fit(X_train, y_train)

# 4. Get the best model
best_dt_model = grid_search.best_estimator_

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV R²: {grid_search.best_score_:.4f}")

# %%
# Evaluate tuned Decision Tree on test set

y_pred_tuned_dt = best_dt_model.predict(X_test)

# 2. Evaluate
r2_tuned_dt = r2_score(y_test, y_pred_tuned_dt)
mae_tuned_dt = mean_absolute_error(np.exp(y_test), np.exp(y_pred_tuned_dt))

print(f"Tuned Decision Tree R-squared: {r2_tuned_dt:.4f}")
print(f"Tuned Decision Tree MAE: {mae_tuned_dt:.2f}")

# %%

# Checking for over or underfitting by comparing train and test scores

train_score = best_dt_model.score(X_train, y_train)
test_score = best_dt_model.score(X_test, y_test)

print(f"Train R-squared: {train_score:.4f}")
print(f"Test R-squared: {test_score:.4f}")

# %% [markdown]
# #### Hyperparameter Tuning — Results
#
# | Metric | Base DT | Tuned DT |
# |--------|---------|----------|
# | R² | 0.8525 | 0.8525 |
# | MAE | ₹115,814 | ₹115,814 |
#
# GridSearchCV returned default values for all parameters (`max_depth: None`, `min_samples_leaf: 1`, `min_samples_split: 2`), meaning no pruning improved performance — the tree in its default form already represents the best configuration for this dataset.
#
# The CV R² of 0.768 vs test R² of 0.853 indicates the model still overfits to some degree. Further improvement would likely require ensemble methods such as Random Forest.

# %%
# Bar chart comparing R² and MAE across models
models = ['Linear Regression', 'Tuned Decision Tree']

r2_scores = [r2_lr, r2_tuned_dt]
mae_scores = [mae_lr, mae_tuned_dt]

# 2. Set up the figure with 2 subplots (R2 on the left, MAE on the right)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
plt.subplots_adjust(wspace=0.3)

bars_r2 = ax1.bar(models, r2_scores, color=['#b3cde3', '#decbe4'])
ax1.set_title('Model Comparison: R-Squared (Quality of Fit)',
              fontsize=14, fontweight='bold')
ax1.set_ylabel('R-Squared Score', fontsize=12)
ax1.set_ylim(0, 1.0)  # Grade scale from 0 to 1

# Add data labels on top of bars
for bar in bars_r2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, height,
             f'{height:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# R² comparison — higher is better
bars_mae = ax2.bar(models, mae_scores, color=[
                   '#fbb4ae', '#b3cde3'])  # Light Red / Light Blue
ax2.set_title('Model Comparison: Mean Absolute Error (Average Error in INR)',
              fontsize=14, fontweight='bold')
ax2.set_ylabel('MAE Score', fontsize=12)
# Add data labels on top of bars
for bar in bars_mae:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2, height,
             f'{height:,.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# %% [markdown]
# The Decision Tree outperforms Linear Regression, achieving a higher R² and significantly lower MAE — confirming its ability to capture the non-linear relationships in this dataset.
#
# However, the gap between train and test R² indicates overfitting. A Random Forest model will be introduced next — by aggregating multiple Decision Trees, it reduces overfitting and is expected to improve generalisation on unseen data.

# %% [markdown]
# ### Improved Model 2 — Random Forest Regressor
#
# A Random Forest is introduced to address the overfitting observed in the single Decision Tree. By training multiple trees on random subsets of the data and averaging their predictions, it reduces variance without significantly increasing bias.
#
# Additional advantages for this dataset:
# - Robust to outliers — relevant given our high-end luxury listings
# - No scaling required — consistent with our preprocessing approach
# - Provides feature importance scores to validate our feature selection decisions
#

# %%


# 1. Initialize the Random Forest
# n_estimators=100 (100 trees)
# n_jobs=-1 uses all your computer's processors for speed
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# 2. Train the model
rf_model.fit(X_train, y_train)

# 3. Predict and Evaluate
y_pred_rf = rf_model.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)

# Remember to np.exp() because we trained on Log Price
actual_prices = np.exp(y_test)
predicted_prices = np.exp(y_pred_rf)

mae_rf = mean_absolute_error(actual_prices, predicted_prices)

print(f"Random Forest R-squared: {r2_rf:.4f}")
print(f"Random Forest MAE: {mae_rf:.2f} ")


# %% [markdown]
# The Random Forest improves on the Decision Tree's R² (0.865 vs 0.853), confirming it captures more of the overall pricing pattern. The train/test gap has also narrowed (0.963 vs 0.865) compared to the Decision Tree (0.977 vs 0.853), indicating less overfitting.
#
# However, the MAE has increased to ₹168,283 — higher than the Decision Tree's ₹115,814. This is likely because the Random Forest averages predictions across 100 trees, which smooths out extreme values and reduces precision on high-end luxury listings.
#
# This presents a tradeoff — the Random Forest generalises better overall but the Decision Tree makes more precise individual predictions. For deployment, the Decision Tree may be more appropriate given the MAE difference.

# %%
# Visualisations of r2 fit and mae for the base random forest tree model

actual = np.exp(y_test)
predicted = np.exp(y_pred_rf)

plt.figure(figsize=(8, 6))
plt.scatter(actual, predicted, alpha=0.7, color='red')

plt.plot(
    [actual.min(), actual.max()],
    [actual.min(), actual.max()], color='black', label='Perfect fit'

)

plt.xlabel("Actual Price (INR)")
plt.ylabel("Predicted Price (INR)")
plt.title("Random Forest: Actual vs Predicted Prices")
plt.tight_layout()
plt.show()

# %% [markdown]
# The scatter plot shows predictions clustering more tightly around the perfect fit line compared to the base Decision Tree — confirming the Random Forest is making more accurate predictions across a wider price range. High-end vehicles remain the most difficult to predict accurately.

# %%
# Checking for over or underfitting by comparing train and test scores

train_score_rf = rf_model.score(X_train, y_train)
test_score_rf = rf_model.score(X_test, y_test)

print(f"Train R-squared: {train_score_rf:.4f}")
print(f"Test R-squared: {test_score_rf:.4f}")

# %% [markdown]
# The Random Forest improves on the Decision Tree's R² (0.865 vs 0.853), confirming it captures more of the overall pricing pattern. The train/test gap has also narrowed (0.963 vs 0.865) compared to the Decision Tree (0.977 vs 0.853), indicating less overfitting.
#
# However, the MAE has increased to ₹168,283 — higher than the Decision Tree's ₹115,814. This is likely because the Random Forest averages predictions across 100 trees, which smooths out extreme values and reduces precision on high-end luxury listings.
#
# This presents a tradeoff — the Random Forest generalises better overall but the Decision Tree makes more precise individual predictions. For deployment, the Decision Tree may be more appropriate given the MAE difference.

# %%

# 1. Define the search space
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'bootstrap': [True, False]
}

# 2. Run the search (n_iter=10 limits it so it doesn't take forever)
rf_random = RandomizedSearchCV(RandomForestRegressor(random_state=42),
                               param_distributions=param_dist,
                               n_iter=10, cv=3, scoring='r2', n_jobs=-1)

rf_random.fit(X_train, y_train)

# 3. Get results
best_rf = rf_random.best_estimator_
y_pred_final = best_rf.predict(X_test)
r2_best_rf = r2_score(y_test, y_pred_final)
mae_best_rf = mean_absolute_error(np.exp(y_test), np.exp(y_pred_final))
print(f"Best Parameters: {rf_random.best_params_}")
print(f"Tuned RF R²: {r2_best_rf:.4f}")
print(f"Tuned RF MAE: ₹{mae_best_rf:,.2f}")

# %% [markdown]
# #### Tuned Random Forest — Results
#
# | Metric | Base RF | Tuned RF |
# |--------|---------|----------|
# | R² | 0.8649 | 0.8626 |
# | MAE | ₹168,283 | ₹171,996 |
#
# Tuning marginally decreased both R² and MAE, meaning the base Random Forest was already near-optimal for this dataset. The base RF will be carried forward as the final Random Forest model.
#
# This mirrors the result seen in the Decision Tree tuning — suggesting the dataset's complexity is better addressed through model architecture than hyperparameter adjustment.

# %%

# Checking for over or underfitting by comparing train and test scores

train_score_best_rf = best_rf.score(X_train, y_train)
test_score_best_rf = best_rf.score(X_test, y_test)

print(f"Train R-squared: {train_score_best_rf:.4f}")
print(f"Test R-squared: {test_score_best_rf:.4f}")

# %% [markdown]
# Although hyperparameter tuning was explored, the base Random Forest outperformed all tuned variants on both R² and MAE. It is therefore selected as the final Random Forest model.
#
# This result, combined with the narrower train/test gap compared to the Decision Tree, confirms the Random Forest as the better generalising model — even if the Decision Tree produces more precise individual predictions.

# %%
'''# 1. Create the new feature
# We add +1 to Age to avoid dividing by zero for brand new cars
X_train['usage_intensity'] = X_train['kmDriven'] / (X_train['Age'] + 1)
X_test['usage_intensity'] = X_test['kmDriven'] / (X_test['Age'] + 1)

# 2. Re-scale everything (now including our new feature)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train the Random Forest
rf_final = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf_final.fit(X_train_scaled, y_train)

# 4. Evaluate
y_pred_final = rf_final.predict(X_test_scaled)
r2_final = r2_score(y_test, y_pred_final)
mae_final = mean_absolute_error(np.exp(y_test), np.exp(y_pred_final))

print(f"Experimental R-squared: {r2_final:.4f}")
print(f"Experimental MAE: {mae_final:,.2f} INR")'''

# %% [markdown]
# # Attempted to engineer a new feature — did not improve model performance and was excluded

# %%
# Final model comparison — R² and MAE across all three models
models = ['Linear Regression', 'Decision Tree', 'Random Forest']
r2_scores = [r2_lr, r2_tuned_dt, r2_rf]
mae_scores = [mae_lr, mae_tuned_dt, mae_rf]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# R² comparison — higher is better
colors1 = ['#ecf0f1', '#aed6f1', '#a9dfbf']
bars1 = ax1.bar(models, r2_scores, color=colors1, edgecolor='black')
ax1.set_title('Model Performance Comparison (R²)',
              fontsize=14, fontweight='bold')
ax1.set_ylim(0, 1.0)
ax1.set_ylabel('R² Score')
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.02,
             f'{yval:.3f}', ha='center', fontweight='bold')

# MAE comparison — lower is better
colors2 = ['#f1948a', '#f9e79f', '#fad7a0']
bars2 = ax2.bar(models, mae_scores, color=colors2, edgecolor='black')
ax2.set_title('Mean Absolute Error — Lower is Better',
              fontsize=14, fontweight='bold')
ax2.set_ylabel('MAE (₹)')
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 5000,
             f'₹{int(yval):,}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Model Selection — Final Decision
#
# | Metric | Decision Tree | Random Forest | Winner |
# |--------|--------------|---------------|--------|
# | R² | 0.853 | 0.865 | RF ✅ |
# | MAE | ₹115,814 | ₹168,283 | DT ✅ |
# | Overfitting gap | 0.12 | 0.09 | RF ✅ |
#
# The Random Forest is selected as the final model, winning on 2 out of 3 metrics. Its higher R² confirms it captures more of the pricing pattern, and its lower overfitting gap means it will generalise more consistently to new, unseen listings in deployment.
#
# The Decision Tree's lower MAE is noted — however, this advantage is largely driven by luxury outlier listings where the Random Forest's averaging behaviour reduces precision. For the majority of budget and mid-range vehicles that make up most real-world listings, the Random Forest is the more robust and reliable choice.
#
# **The higher MAE is an acknowledged tradeoff**, not a fundamental weakness — and will be flagged as a potential improvement in the conclusion.
#
#

# %%
# Extract and plot top 10 most influential features from the final Random Forest model
importances = rf_model.feature_importances_
feature_names = X_train.columns

# Organise and sort by importance
feature_importance_df = pd.DataFrame(
    {'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(
    by='Importance', ascending=False).head(10)

# Plot feature importances
plt.figure(figsize=(12, 7))
sns.barplot(
    data=feature_importance_df,
    x='Importance',
    y='Feature',
    hue='Feature',
    palette='magma',
    legend=False
)
plt.title('Top 10 Most Influential Features on Car Price',
          fontsize=15, fontweight='bold', pad=20)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# The feature importance analysis confirms that `model_encoded`, `Age`, and `kmDriven` are the strongest individual predictors. Notably, `Transmission`, `Brand_Grouped` and `FuelType` each contribute multiple influential categories — confirming that all retained features carry meaningful pricing signal despite appearing lower individually.

# %% [markdown]
# ## 8. Insights & Conclusion
#
# ### Summary
# This project built and evaluated three regression models to predict used car prices based on listing features. The Random Forest was selected as the final model with an R² of 0.865 and MAE of ₹168,283.
#
# ### Key Insights
# - `Age`, `kmDriven` and `model_encoded` are the strongest predictors of used car price
# - The relationship between features and price is non-linear — tree-based models significantly outperform Linear Regression
# - Luxury vehicles remain the hardest to predict accurately due to high price variance and limited listings
#
# ### Limitations & Future Improvements
# - The `AdditionalInfo` column was dropped — keyword extraction could recover valuable features
# - Ensemble methods like Gradient Boosting (XGBoost) would likely reduce the MAE further
# - A larger, more balanced dataset with more luxury listings would improve high-end predictions
# - Mandatory data fields at listing stage would improve future model retraining
#
# ### Final Model
# | Model | R² | MAE |
# |-------|----|-----|
# | Linear Regression | 0.667 | ₹318,690 |
# | Decision Tree | 0.853 | ₹115,814 |
# | **Random Forest** | **0.865** | **₹168,283** |

# %%
print(df_final['model'].unique())
print(df_final['Transmission'].unique())
print(df_final['Brand_Grouped'].unique())
print(df_final['FuelType'].unique())
print(df_final['Owner'].unique())

# %%
joblib.dump(rf_model, 'car_price_model.pkl')
joblib.dump(model_means, 'model_means.pkl')
joblib.dump(X_train.columns.tolist(), 'model_columns.pkl')

# %% [markdown]
# git lfs install
# git lfs track "car_price_model.pkl"
