# 🛍️ RETAIL & MARKETING ANALYTICS PROJECT
## End-to-End Data Analytics Project for Students

---

## 📋 TABLE OF CONTENTS
1. [Project Overview](#project-overview)
2. [Business Context](#business-context)
3. [Project Objectives](#project-objectives)
4. [Technical Stack](#technical-stack)
5. [Data Acquisition](#data-acquisition)
6. [Data Cleaning & Preprocessing](#data-cleaning--preprocessing)
7. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
8. [Advanced Analytics](#advanced-analytics)
9. [Customer Segmentation (Clustering)](#customer-segmentation-clustering)
10. [KPI Design & Metrics](#kpi-design--metrics)
11. [Dashboard Creation](#dashboard-creation)
12. [Insights & Recommendations](#insights--recommendations)
13. [GitHub Repository Setup](#github-repository-setup)
14. [Executive Summary](#executive-summary)
15. [Presentation Slides Overview](#presentation-slides-overview)

---

## 1. PROJECT OVERVIEW

### **Project Title:** Retail & Marketing Analytics: Customer Segmentation & Sales Optimization

### **Domain:** Retail, E-commerce, Marketing Analytics

### **Target Roles:** 
- Marketing Analyst
- Business Intelligence Analyst
- Data Analyst
- Retail Analytics Specialist

### **Project Duration:** 3-4 Weeks

### **Difficulty Level:** Intermediate to Advanced

---

## 2. BUSINESS CONTEXT

### **Scenario:**
You are a **Marketing Analyst** at a multi-channel retail company that sells various products across different regions. The company has been experiencing:
- Declining customer retention rates
- Inconsistent sales performance across product categories
- Inefficient marketing spend
- Limited understanding of customer behavior

### **Business Problem:**
The executive team needs data-driven insights to:
- Identify high-value customer segments
- Optimize marketing campaigns
- Improve inventory management
- Increase customer lifetime value (CLV)
- Reduce customer acquisition costs (CAC)

---

## 3. PROJECT OBJECTIVES

### **Primary Objectives:**
1. **Customer Segmentation:** Use clustering algorithms to identify distinct customer segments
2. **Sales Performance Analysis:** Analyze sales trends, seasonality, and product performance
3. **Marketing Effectiveness:** Evaluate campaign performance and ROI
4. **KPI Dashboard:** Design and build an interactive dashboard for stakeholders
5. **Actionable Insights:** Provide data-driven recommendations

### **Success Metrics:**
- Identification of 3-5 distinct customer segments
- 15+ KPIs tracked and visualized
- Interactive dashboard with drill-down capabilities
- Actionable recommendations with projected impact

---

## 4. TECHNICAL STACK

### **Programming & Analysis:**
- **Python 3.9+**
  - pandas, numpy (data manipulation)
  - matplotlib, seaborn, plotly (visualization)
  - scikit-learn (machine learning & clustering)
  - scipy (statistical analysis)

### **Dashboard Tools:**
- **Power BI** (Primary - recommended for students)
- **Tableau Public** (Alternative)
- **Python Dash** (For interactive web apps)

### **Version Control:**
- **Git & GitHub**

### **Additional Tools:**
- Jupyter Notebook / VS Code
- Excel (initial data review)

---

## 5. DATA ACQUISITION

### **Step 1: Download Dataset from Kaggle**

**Recommended Dataset:** [Retail Sales Dataset with Seasonal Trends & Marketing](https://www.kaggle.com/datasets/abdullah0a/retail-sales-data-with-seasonal-trends-and-marketing)

**Alternative Datasets:**
- [Retail Transactions Dataset](https://www.kaggle.com/datasets/prasad22/retail-transactions-dataset)
- [E-commerce Sales Data 2024](https://www.kaggle.com/datasets/datascientist97/e-commerece-sales-data-2024)

### **Step 2: Setup Project Structure**

```bash
# Create project directory
mkdir retail-marketing-analytics
cd retail-marketing-analytics

# Create folder structure
mkdir data
mkdir notebooks
mkdir scripts
mkdir dashboards
mkdir outputs
mkdir docs

# Create subdirectories
mkdir data/raw
mkdir data/processed
mkdir outputs/figures
mkdir outputs/reports
```

### **Step 3: Download Data via Kaggle API**

```python
# Install Kaggle API
!pip install kaggle

# Setup Kaggle credentials (place kaggle.json in ~/.kaggle/)
# Download dataset
!kaggle datasets download -d abdullah0a/retail-sales-data-with-seasonal-trends-and-marketing

# Unzip
import zipfile
with zipfile.ZipFile('retail-sales-data-with-seasonal-trends-and-marketing.zip', 'r') as zip_ref:
    zip_ref.extractall('data/raw/')
```

### **Step 4: Load Data**

```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df_sales = pd.read_csv('data/raw/retail_sales_data.csv')

# Initial inspection
print("Dataset Shape:", df_sales.shape)
print("\nFirst 5 rows:")
print(df_sales.head())
print("\nData Types:")
print(df_sales.dtypes)
print("\nBasic Statistics:")
print(df_sales.describe())
```

---

## 6. DATA CLEANING & PREPROCESSING

### **Step 1: Data Quality Assessment**

```python
# Check for missing values
print("Missing Values:")
print(df_sales.isnull().sum())
print("\nMissing Value Percentage:")
print((df_sales.isnull().sum() / len(df_sales)) * 100)

# Check for duplicates
duplicates = df_sales.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates}")

# Check data types
print("\nData Type Issues:")
for col in df_sales.columns:
    print(f"{col}: {df_sales[col].dtype}")
```

### **Step 2: Handle Missing Values**

```python
# Strategy for handling missing values
# Numerical columns: fill with median/mean
# Categorical columns: fill with mode or 'Unknown'

# Example:
numerical_cols = df_sales.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df_sales.select_dtypes(include=['object']).columns

# Fill numerical missing values
for col in numerical_cols:
    if df_sales[col].isnull().sum() > 0:
        df_sales[col].fillna(df_sales[col].median(), inplace=True)

# Fill categorical missing values
for col in categorical_cols:
    if df_sales[col].isnull().sum() > 0:
        df_sales[col].fillna(df_sales[col].mode()[0], inplace=True)

# Verify
print("Missing values after treatment:", df_sales.isnull().sum().sum())
```

### **Step 3: Remove Duplicates**

```python
# Remove exact duplicates
df_sales = df_sales.drop_duplicates()
print(f"Shape after removing duplicates: {df_sales.shape}")
```

### **Step 4: Data Type Conversion**

```python
# Convert date columns to datetime
date_columns = ['Order_Date', 'Ship_Date']  # Adjust based on your dataset
for col in date_columns:
    if col in df_sales.columns:
        df_sales[col] = pd.to_datetime(df_sales[col], errors='coerce')

# Convert categorical columns
categorical_columns = ['Customer_Segment', 'Product_Category', 'Region']
for col in categorical_columns:
    if col in df_sales.columns:
        df_sales[col] = df_sales[col].astype('category')
```

### **Step 5: Outlier Detection & Treatment**

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Identify outliers using IQR method
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Check for outliers in Sales column
if 'Sales' in df_sales.columns:
    outliers, lb, ub = detect_outliers_iqr(df_sales, 'Sales')
    print(f"Outliers in Sales: {len(outliers)}")
    print(f"Lower Bound: {lb}, Upper Bound: {ub}")
    
    # Visualize
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.boxplot(df_sales['Sales'])
    plt.title('Sales Distribution (Before Treatment)')
    plt.ylabel('Sales')
    
    # Cap outliers instead of removing
    df_sales['Sales_Capped'] = df_sales['Sales'].clip(lower=lb, upper=ub)
    
    plt.subplot(1, 2, 2)
    plt.boxplot(df_sales['Sales_Capped'])
    plt.title('Sales Distribution (After Treatment)')
    plt.ylabel('Sales')
    plt.tight_layout()
    plt.savefig('outputs/figures/outlier_treatment.png', dpi=300, bbox_inches='tight')
    plt.show()
```

### **Step 6: Feature Engineering**

```python
# Create time-based features
if 'Order_Date' in df_sales.columns:
    df_sales['Year'] = df_sales['Order_Date'].dt.year
    df_sales['Month'] = df_sales['Order_Date'].dt.month
    df_sales['Quarter'] = df_sales['Order_Date'].dt.quarter
    df_sales['Day_of_Week'] = df_sales['Order_Date'].dt.dayofweek
    df_sales['Week_of_Year'] = df_sales['Order_Date'].dt.isocalendar().week
    df_sales['Is_Weekend'] = df_sales['Day_of_Week'].isin([5, 6]).astype(int)

# Create revenue metrics
if 'Quantity' in df_sales.columns and 'Unit_Price' in df_sales.columns:
    df_sales['Revenue'] = df_sales['Quantity'] * df_sales['Unit_Price']
    df_sales['Discount_Amount'] = df_sales['Unit_Price'] * df_sales['Discount'] / 100
    df_sales['Net_Revenue'] = df_sales['Revenue'] - df_sales['Discount_Amount']

# Create customer metrics
if 'Customer_ID' in df_sales.columns:
    customer_stats = df_sales.groupby('Customer_ID').agg({
        'Order_ID': 'count',  # Number of orders
        'Revenue': 'sum',      # Total revenue
        'Order_Date': ['min', 'max']  # First and last purchase
    }).reset_index()
    
    customer_stats.columns = ['Customer_ID', 'Order_Count', 'Total_Revenue', 
                              'First_Purchase', 'Last_Purchase']
    
    # Calculate recency
    customer_stats['Recency_Days'] = (pd.to_datetime('today') - 
                                       customer_stats['Last_Purchase']).dt.days
    
    # Merge back to main dataframe
    df_sales = df_sales.merge(customer_stats[['Customer_ID', 'Order_Count', 
                                                'Total_Revenue', 'Recency_Days']], 
                              on='Customer_ID', how='left')

# Create product performance metrics
if 'Product_ID' in df_sales.columns:
    product_stats = df_sales.groupby('Product_ID').agg({
        'Revenue': ['sum', 'mean'],
        'Quantity': 'sum'
    }).reset_index()
    product_stats.columns = ['Product_ID', 'Product_Total_Revenue', 
                             'Product_Avg_Revenue', 'Product_Total_Quantity']
    df_sales = df_sales.merge(product_stats, on='Product_ID', how='left')

print("Feature Engineering Completed!")
print(f"New DataFrame Shape: {df_sales.shape}")
```

### **Step 7: Save Cleaned Data**

```python
# Save cleaned dataset
df_sales.to_csv('data/processed/cleaned_retail_sales.csv', index=False)
print("Cleaned data saved successfully!")

# Create data dictionary
data_dict = pd.DataFrame({
    'Column_Name': df_sales.columns,
    'Data_Type': df_sales.dtypes.values,
    'Non_Null_Count': df_sales.count().values,
    'Null_Count': df_sales.isnull().sum().values,
    'Unique_Values': [df_sales[col].nunique() for col in df_sales.columns]
})
data_dict.to_csv('docs/data_dictionary.csv', index=False)
```

---

## 7. EXPLORATORY DATA ANALYSIS (EDA)

### **Step 1: Univariate Analysis**

```python
import plotly.express as px
import plotly.graph_objects as go

# Numerical columns distribution
numerical_cols = df_sales.select_dtypes(include=['float64', 'int64']).columns

fig, axes = plt.subplots(len(numerical_cols)//3 + 1, 3, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    axes[i].hist(df_sales[col], bins=30, edgecolor='black', alpha=0.7)
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('outputs/figures/numerical_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# Categorical columns distribution
categorical_cols = df_sales.select_dtypes(include=['object', 'category']).columns

for col in categorical_cols[:5]:  # Top 5 categorical
    fig = px.bar(df_sales[col].value_counts().reset_index(), 
                 x='index', y=col, 
                 title=f'Distribution of {col}',
                 labels={'index': col, col: 'Count'})
    fig.write_html(f'outputs/figures/{col}_distribution.html')
```

### **Step 2: Bivariate Analysis**

```python
# Sales by Category
if 'Product_Category' in df_sales.columns and 'Revenue' in df_sales.columns:
    category_sales = df_sales.groupby('Product_Category')['Revenue'].sum().sort_values(ascending=False)
    
    fig = px.bar(category_sales.reset_index(), 
                 x='Product_Category', y='Revenue',
                 title='Total Revenue by Product Category',
                 labels={'Revenue': 'Total Revenue ($)', 'Product_Category': 'Category'},
                 color='Revenue',
                 color_continuous_scale='viridis')
    fig.write_html('outputs/figures/revenue_by_category.html')

# Sales by Region
if 'Region' in df_sales.columns:
    region_sales = df_sales.groupby('Region')['Revenue'].sum().sort_values(ascending=False)
    
    fig = px.pie(region_sales.reset_index(), 
                 values='Revenue', names='Region',
                 title='Revenue Distribution by Region',
                 hole=0.4)
    fig.write_html('outputs/figures/revenue_by_region.html')

# Correlation heatmap
numerical_data = df_sales.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numerical_data.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.savefig('outputs/figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
```

### **Step 3: Time Series Analysis**

```python
# Monthly sales trend
if 'Order_Date' in df_sales.columns:
    df_sales['YearMonth'] = df_sales['Order_Date'].dt.to_period('M')
    monthly_sales = df_sales.groupby('YearMonth')['Revenue'].sum().reset_index()
    monthly_sales['YearMonth'] = monthly_sales['YearMonth'].astype(str)
    
    fig = px.line(monthly_sales, x='YearMonth', y='Revenue',
                  title='Monthly Revenue Trend',
                  markers=True)
    fig.update_xaxis(tickangle=45)
    fig.write_html('outputs/figures/monthly_revenue_trend.html')

# Quarterly comparison
    quarterly_sales = df_sales.groupby(['Year', 'Quarter'])['Revenue'].sum().reset_index()
    
    fig = px.bar(quarterly_sales, x='Quarter', y='Revenue', color='Year',
                 title='Quarterly Revenue Comparison',
                 barmode='group')
    fig.write_html('outputs/figures/quarterly_comparison.html')

# Day of week analysis
    dow_sales = df_sales.groupby('Day_of_Week')['Revenue'].mean().reset_index()
    dow_sales['Day_Name'] = dow_sales['Day_of_Week'].map({
        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
        4: 'Friday', 5: 'Saturday', 6: 'Sunday'
    })
    
    fig = px.bar(dow_sales, x='Day_Name', y='Revenue',
                 title='Average Revenue by Day of Week',
                 color='Revenue',
                 color_continuous_scale='blues')
    fig.write_html('outputs/figures/revenue_by_day.html')
```

### **Step 4: Customer Behavior Analysis**

```python
# Customer purchase frequency
customer_frequency = df_sales.groupby('Customer_ID').size().reset_index(name='Purchase_Count')

plt.figure(figsize=(10, 6))
plt.hist(customer_frequency['Purchase_Count'], bins=50, edgecolor='black', alpha=0.7)
plt.title('Customer Purchase Frequency Distribution')
plt.xlabel('Number of Purchases')
plt.ylabel('Number of Customers')
plt.axvline(customer_frequency['Purchase_Count'].mean(), color='red', 
            linestyle='--', label=f'Mean: {customer_frequency["Purchase_Count"].mean():.2f}')
plt.legend()
plt.savefig('outputs/figures/customer_frequency.png', dpi=300, bbox_inches='tight')
plt.show()

# Top 10 customers by revenue
top_customers = df_sales.groupby('Customer_ID')['Revenue'].sum().sort_values(ascending=False).head(10)

fig = px.bar(top_customers.reset_index(), x='Customer_ID', y='Revenue',
             title='Top 10 Customers by Revenue',
             labels={'Revenue': 'Total Revenue ($)', 'Customer_ID': 'Customer ID'})
fig.write_html('outputs/figures/top_customers.html')
```

### **Step 5: Product Performance Analysis**

```python
# Top selling products
top_products = df_sales.groupby('Product_ID').agg({
    'Quantity': 'sum',
    'Revenue': 'sum'
}).sort_values('Revenue', ascending=False).head(20)

fig = go.Figure()
fig.add_trace(go.Bar(x=top_products.index, y=top_products['Revenue'], 
                     name='Revenue', yaxis='y', marker_color='indianred'))
fig.add_trace(go.Scatter(x=top_products.index, y=top_products['Quantity'], 
                         name='Quantity', yaxis='y2', mode='lines+markers',
                         marker_color='lightsalmon'))

fig.update_layout(
    title='Top 20 Products: Revenue vs Quantity Sold',
    xaxis=dict(title='Product ID'),
    yaxis=dict(title='Revenue ($)', side='left'),
    yaxis2=dict(title='Quantity', overlaying='y', side='right'),
    hovermode='x unified'
)
fig.write_html('outputs/figures/top_products_analysis.html')

# Product category profitability
category_metrics = df_sales.groupby('Product_Category').agg({
    'Revenue': 'sum',
    'Quantity': 'sum',
    'Order_ID': 'count'
}).reset_index()
category_metrics['Avg_Order_Value'] = category_metrics['Revenue'] / category_metrics['Order_ID']

fig = px.scatter(category_metrics, x='Order_ID', y='Avg_Order_Value',
                 size='Revenue', color='Product_Category',
                 title='Product Category Performance: Orders vs AOV',
                 labels={'Order_ID': 'Number of Orders', 
                         'Avg_Order_Value': 'Average Order Value ($)'})
fig.write_html('outputs/figures/category_performance.html')
```

---

## 8. ADVANCED ANALYTICS

### **Step 1: RFM Analysis (Recency, Frequency, Monetary)**

```python
from datetime import datetime

# Calculate RFM metrics
analysis_date = df_sales['Order_Date'].max() + pd.Timedelta(days=1)

rfm = df_sales.groupby('Customer_ID').agg({
    'Order_Date': lambda x: (analysis_date - x.max()).days,  # Recency
    'Order_ID': 'count',  # Frequency
    'Revenue': 'sum'  # Monetary
}).reset_index()

rfm.columns = ['Customer_ID', 'Recency', 'Frequency', 'Monetary']

# Create RFM scores
rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

# Combined RFM score
rfm['RFM_Score'] = (rfm['R_Score'].astype(str) + 
                    rfm['F_Score'].astype(str) + 
                    rfm['M_Score'].astype(str))

# Customer segmentation based on RFM
def segment_customers(row):
    if row['R_Score'] >= 4 and row['F_Score'] >= 4 and row['M_Score'] >= 4:
        return 'Champions'
    elif row['R_Score'] >= 3 and row['F_Score'] >= 3:
        return 'Loyal Customers'
    elif row['R_Score'] >= 4 and row['F_Score'] <= 2:
        return 'Potential Loyalists'
    elif row['R_Score'] >= 3 and row['F_Score'] <= 2 and row['M_Score'] >= 3:
        return 'New Customers'
    elif row['R_Score'] <= 2 and row['F_Score'] >= 3:
        return 'At Risk'
    elif row['R_Score'] <= 2 and row['F_Score'] <= 2:
        return 'Lost Customers'
    else:
        return 'Others'

rfm['Customer_Segment'] = rfm.apply(segment_customers, axis=1)

# Visualize RFM segments
segment_counts = rfm['Customer_Segment'].value_counts()

fig = px.pie(segment_counts.reset_index(), values='count', names='Customer_Segment',
             title='Customer Distribution by RFM Segments',
             hole=0.4,
             color_discrete_sequence=px.colors.qualitative.Set3)
fig.write_html('outputs/figures/rfm_segments.html')

# RFM summary by segment
rfm_summary = rfm.groupby('Customer_Segment').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'Customer_ID': 'count'
}).round(2)
rfm_summary.columns = ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Customer_Count']
print("\nRFM Summary by Segment:")
print(rfm_summary)

# Save RFM analysis
rfm.to_csv('data/processed/rfm_analysis.csv', index=False)
```

### **Step 2: Market Basket Analysis**

```python
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Prepare transaction data
transactions = df_sales.groupby('Order_ID')['Product_ID'].apply(list).tolist()

# Encode transactions
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Find frequent itemsets
frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules = rules.sort_values('lift', ascending=False).head(20)

print("\nTop 20 Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Visualize top rules
fig = px.scatter(rules, x='support', y='confidence', 
                 size='lift', color='lift',
                 title='Association Rules: Support vs Confidence',
                 labels={'support': 'Support', 'confidence': 'Confidence'},
                 color_continuous_scale='viridis')
fig.write_html('outputs/figures/association_rules.html')

# Save rules
rules.to_csv('outputs/reports/market_basket_rules.csv', index=False)
```

### **Step 3: Cohort Analysis**

```python
# Create cohort data
df_sales['Order_Month'] = df_sales['Order_Date'].dt.to_period('M')
df_sales['Cohort'] = df_sales.groupby('Customer_ID')['Order_Date'].transform('min').dt.to_period('M')

# Calculate cohort index
def get_cohort_period(df):
    df['Cohort_Index'] = (df['Order_Month'] - df['Cohort']).apply(lambda x: x.n)
    return df

df_cohort = get_cohort_period(df_sales)

# Cohort size
cohort_size = df_cohort.groupby('Cohort')['Customer_ID'].nunique()

# Retention matrix
retention = df_cohort.groupby(['Cohort', 'Cohort_Index'])['Customer_ID'].nunique().reset_index()
retention_matrix = retention.pivot(index='Cohort', columns='Cohort_Index', values='Customer_ID')

# Calculate retention rate
retention_rate = retention_matrix.divide(cohort_size, axis=0) * 100

# Visualize cohort retention
plt.figure(figsize=(14, 8))
sns.heatmap(retention_rate, annot=True, fmt='.1f', cmap='YlGnBu', 
            cbar_kws={'label': 'Retention Rate (%)'})
plt.title('Cohort Retention Analysis')
plt.xlabel('Cohort Index (Months)')
plt.ylabel('Cohort Month')
plt.tight_layout()
plt.savefig('outputs/figures/cohort_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Save cohort data
retention_rate.to_csv('outputs/reports/cohort_retention.csv')
```

### **Step 4: Customer Lifetime Value (CLV) Calculation**

```python
# Calculate CLV components
customer_metrics = df_sales.groupby('Customer_ID').agg({
    'Revenue': 'sum',
    'Order_ID': 'count',
    'Order_Date': ['min', 'max']
}).reset_index()

customer_metrics.columns = ['Customer_ID', 'Total_Revenue', 'Order_Count', 
                            'First_Purchase', 'Last_Purchase']

# Calculate customer lifespan in days
customer_metrics['Lifespan_Days'] = (customer_metrics['Last_Purchase'] - 
                                      customer_metrics['First_Purchase']).dt.days
customer_metrics['Lifespan_Days'] = customer_metrics['Lifespan_Days'].replace(0, 1)

# Calculate metrics
customer_metrics['Avg_Order_Value'] = customer_metrics['Total_Revenue'] / customer_metrics['Order_Count']
customer_metrics['Purchase_Frequency'] = customer_metrics['Order_Count'] / (customer_metrics['Lifespan_Days'] / 365)

# Simple CLV calculation (can be enhanced with churn prediction)
customer_metrics['CLV_Simple'] = (customer_metrics['Avg_Order_Value'] * 
                                   customer_metrics['Purchase_Frequency'] * 
                                   3)  # Assuming 3-year customer lifespan

# Categorize customers by CLV
customer_metrics['CLV_Category'] = pd.qcut(customer_metrics['CLV_Simple'], 
                                            q=4, 
                                            labels=['Low', 'Medium', 'High', 'Very High'])

# Visualize CLV distribution
fig = px.histogram(customer_metrics, x='CLV_Simple', 
                   title='Customer Lifetime Value Distribution',
                   nbins=50,
                   labels={'CLV_Simple': 'CLV ($)'})
fig.write_html('outputs/figures/clv_distribution.html')

# CLV by category
clv_summary = customer_metrics.groupby('CLV_Category').agg({
    'Customer_ID': 'count',
    'CLV_Simple': 'mean',
    'Total_Revenue': 'sum'
}).round(2)

print("\nCLV Summary by Category:")
print(clv_summary)

# Save CLV data
customer_metrics.to_csv('data/processed/customer_clv.csv', index=False)
```

---

## 9. CUSTOMER SEGMENTATION (CLUSTERING)

### **Step 1: Prepare Data for Clustering**

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Select features for clustering
clustering_features = ['Recency', 'Frequency', 'Monetary']
X = rfm[clustering_features].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Features for Clustering:")
print(clustering_features)
print(f"\nScaled Data Shape: {X_scaled.shape}")
```

### **Step 2: Determine Optimal Number of Clusters**

```python
# Elbow method
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot elbow curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')
axes[0].grid(True)

axes[1].plot(K_range, silhouette_scores, 'ro-')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Analysis')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('outputs/figures/optimal_clusters.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nSilhouette Scores by Number of Clusters:")
for k, score in zip(K_range, silhouette_scores):
    print(f"k={k}: {score:.4f}")
```

### **Step 3: Apply K-Means Clustering**

```python
# Choose optimal k (e.g., k=4 based on analysis)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(X_scaled)

print(f"\nClustering completed with k={optimal_k}")
print(f"Silhouette Score: {silhouette_score(X_scaled, rfm['Cluster']):.4f}")
```

### **Step 4: Analyze Clusters**

```python
# Cluster statistics
cluster_summary = rfm.groupby('Cluster').agg({
    'Recency': ['mean', 'std'],
    'Frequency': ['mean', 'std'],
    'Monetary': ['mean', 'std'],
    'Customer_ID': 'count'
}).round(2)

cluster_summary.columns = ['_'.join(col).strip() for col in cluster_summary.columns.values]
cluster_summary = cluster_summary.rename(columns={'Customer_ID_count': 'Customer_Count'})

print("\nCluster Summary Statistics:")
print(cluster_summary)

# Name clusters based on characteristics
def name_cluster(row):
    if row['Recency_mean'] < 50 and row['Frequency_mean'] > 5 and row['Monetary_mean'] > 1000:
        return 'VIP Customers'
    elif row['Frequency_mean'] > 3 and row['Monetary_mean'] > 500:
        return 'High Value'
    elif row['Recency_mean'] < 100 and row['Frequency_mean'] <= 3:
        return 'Potential'
    else:
        return 'At Risk'

cluster_names = cluster_summary.apply(name_cluster, axis=1)
cluster_mapping = dict(enumerate(cluster_names))
rfm['Cluster_Name'] = rfm['Cluster'].map(cluster_mapping)

print("\nCluster Names:")
print(cluster_mapping)

# Save clustered data
rfm.to_csv('data/processed/customer_segments.csv', index=False)
```

### **Step 5: Visualize Clusters**

```python
# 2D visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

rfm['PCA1'] = X_pca[:, 0]
rfm['PCA2'] = X_pca[:, 1]

fig = px.scatter(rfm, x='PCA1', y='PCA2', color='Cluster_Name',
                 title='Customer Segments Visualization (PCA)',
                 labels={'PCA1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
                         'PCA2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)'},
                 hover_data=['Recency', 'Frequency', 'Monetary'])
fig.write_html('outputs/figures/customer_segments_pca.html')

# 3D scatter plot
fig = px.scatter_3d(rfm, x='Recency', y='Frequency', z='Monetary',
                    color='Cluster_Name',
                    title='Customer Segments in RFM Space',
                    labels={'Recency': 'Recency (days)', 
                            'Frequency': 'Frequency (orders)',
                            'Monetary': 'Monetary ($)'})
fig.write_html('outputs/figures/customer_segments_3d.html')

# Cluster size distribution
cluster_counts = rfm['Cluster_Name'].value_counts()

fig = px.bar(cluster_counts.reset_index(), x='Cluster_Name', y='count',
             title='Customer Distribution by Segment',
             labels={'count': 'Number of Customers', 'Cluster_Name': 'Segment'},
             color='count',
             color_continuous_scale='viridis')
fig.write_html('outputs/figures/segment_distribution.html')
```

### **Step 6: Cluster Profiling**

```python
# Detailed profiling report
for cluster_id, cluster_name in cluster_mapping.items():
    print(f"\n{'='*60}")
    print(f"CLUSTER {cluster_id}: {cluster_name}")
    print(f"{'='*60}")
    
    cluster_data = rfm[rfm['Cluster'] == cluster_id]
    
    print(f"\nSize: {len(cluster_data)} customers ({len(cluster_data)/len(rfm)*100:.1f}%)")
    print(f"\nRFM Metrics:")
    print(f"  - Avg Recency: {cluster_data['Recency'].mean():.1f} days")
    print(f"  - Avg Frequency: {cluster_data['Frequency'].mean():.1f} orders")
    print(f"  - Avg Monetary: ${cluster_data['Monetary'].mean():.2f}")
    print(f"  - Total Revenue: ${cluster_data['Monetary'].sum():.2f}")
    
    print(f"\nRecommended Actions:")
    if cluster_name == 'VIP Customers':
        print("  ✓ Exclusive rewards and early access to new products")
        print("  ✓ Personalized communication and dedicated support")
        print("  ✓ Premium tier membership programs")
    elif cluster_name == 'High Value':
        print("  ✓ Loyalty programs and referral incentives")
        print("  ✓ Cross-sell and upsell campaigns")
        print("  ✓ Regular engagement through email marketing")
    elif cluster_name == 'Potential':
        print("  ✓ Nurture with targeted promotions")
        print("  ✓ Product recommendations based on browsing history")
        print("  ✓ Time-limited offers to encourage repeat purchase")
    else:
        print("  ✓ Win-back campaigns with special discounts")
        print("  ✓ Survey to understand reasons for churn")
        print("  ✓ Re-engagement through personalized content")

# Save profiling report
with open('outputs/reports/cluster_profiling.txt', 'w') as f:
    for cluster_id, cluster_name in cluster_mapping.items():
        f.write(f"\n{'='*60}\n")
        f.write(f"CLUSTER {cluster_id}: {cluster_name}\n")
        f.write(f"{'='*60}\n")
        cluster_data = rfm[rfm['Cluster'] == cluster_id]
        f.write(f"\nSize: {len(cluster_data)} customers ({len(cluster_data)/len(rfm)*100:.1f}%)\n")
        f.write(f"\nAvg Recency: {cluster_data['Recency'].mean():.1f} days\n")
        f.write(f"Avg Frequency: {cluster_data['Frequency'].mean():.1f} orders\n")
        f.write(f"Avg Monetary: ${cluster_data['Monetary'].mean():.2f}\n")
        f.write(f"Total Revenue: ${cluster_data['Monetary'].sum():.2f}\n\n")
```

---

## 10. KPI DESIGN & METRICS

### **Step 1: Define Key Performance Indicators**

```python
# Calculate comprehensive KPIs
kpis = {}

# Revenue KPIs
kpis['Total_Revenue'] = df_sales['Revenue'].sum()
kpis['Avg_Order_Value'] = df_sales.groupby('Order_ID')['Revenue'].sum().mean()
kpis['Revenue_Per_Customer'] = df_sales.groupby('Customer_ID')['Revenue'].sum().mean()

# Customer KPIs
kpis['Total_Customers'] = df_sales['Customer_ID'].nunique()
kpis['Total_Orders'] = df_sales['Order_ID'].nunique()
kpis['Avg_Orders_Per_Customer'] = kpis['Total_Orders'] / kpis['Total_Customers']
kpis['Customer_Retention_Rate'] = (rfm[rfm['Frequency'] > 1].shape[0] / kpis['Total_Customers']) * 100

# Product KPIs
kpis['Total_Products_Sold'] = df_sales['Quantity'].sum()
kpis['Avg_Items_Per_Order'] = df_sales.groupby('Order_ID')['Quantity'].sum().mean()
kpis['Total_SKUs'] = df_sales['Product_ID'].nunique()

# Time-based KPIs
kpis['Avg_Purchase_Frequency_Days'] = rfm['Recency'].mean()
kpis['Avg_Customer_Lifespan_Days'] = customer_metrics['Lifespan_Days'].mean()

# Profitability KPIs (assuming profit margin)
profit_margin = 0.25  # 25% profit margin assumption
kpis['Total_Profit'] = kpis['Total_Revenue'] * profit_margin
kpis['Profit_Per_Customer'] = kpis['Total_Profit'] / kpis['Total_Customers']

# Marketing KPIs
kpis['Customer_Acquisition_Cost'] = 50  # Assumption
kpis['Customer_Lifetime_Value'] = customer_metrics['CLV_Simple'].mean()
kpis['CLV_to_CAC_Ratio'] = kpis['Customer_Lifetime_Value'] / kpis['Customer_Acquisition_Cost']

# Conversion & Engagement KPIs
kpis['Repeat_Purchase_Rate'] = (rfm[rfm['Frequency'] > 1].shape[0] / kpis['Total_Customers']) * 100
kpis['Churn_Rate'] = (rfm[rfm['Recency'] > 180].shape[0] / kpis['Total_Customers']) * 100

# Print KPI Dashboard
print("\n" + "="*70)
print("KEY PERFORMANCE INDICATORS (KPIs) DASHBOARD")
print("="*70)

print("\n📊 REVENUE METRICS")
print(f"  Total Revenue: ${kpis['Total_Revenue']:,.2f}")
print(f"  Average Order Value: ${kpis['Avg_Order_Value']:,.2f}")
print(f"  Revenue Per Customer: ${kpis['Revenue_Per_Customer']:,.2f}")
print(f"  Total Profit (Est.): ${kpis['Total_Profit']:,.2f}")

print("\n👥 CUSTOMER METRICS")
print(f"  Total Customers: {kpis['Total_Customers']:,}")
print(f"  Total Orders: {kpis['Total_Orders']:,}")
print(f"  Avg Orders Per Customer: {kpis['Avg_Orders_Per_Customer']:.2f}")
print(f"  Customer Retention Rate: {kpis['Customer_Retention_Rate']:.2f}%")
print(f"  Repeat Purchase Rate: {kpis['Repeat_Purchase_Rate']:.2f}%")
print(f"  Churn Rate: {kpis['Churn_Rate']:.2f}%")

print("\n🛍️ PRODUCT METRICS")
print(f"  Total Products Sold: {kpis['Total_Products_Sold']:,} units")
print(f"  Average Items Per Order: {kpis['Avg_Items_Per_Order']:.2f}")
print(f"  Total SKUs: {kpis['Total_SKUs']:,}")

print("\n💰 CUSTOMER VALUE METRICS")
print(f"  Customer Lifetime Value: ${kpis['Customer_Lifetime_Value']:,.2f}")
print(f"  Customer Acquisition Cost: ${kpis['Customer_Acquisition_Cost']:,.2f}")
print(f"  CLV to CAC Ratio: {kpis['CLV_to_CAC_Ratio']:.2f}x")
print(f"  Profit Per Customer: ${kpis['Profit_Per_Customer']:,.2f}")

print("\n⏱️ TIME-BASED METRICS")
print(f"  Avg Purchase Frequency: {kpis['Avg_Purchase_Frequency_Days']:.1f} days")
print(f"  Avg Customer Lifespan: {kpis['Avg_Customer_Lifespan_Days']:.1f} days")

print("\n" + "="*70)

# Save KPIs to file
kpi_df = pd.DataFrame(list(kpis.items()), columns=['KPI', 'Value'])
kpi_df.to_csv('outputs/reports/kpi_summary.csv', index=False)
```

### **Step 2: Create KPI Tracking Dashboard Data**

```python
# Monthly KPI trends
monthly_kpis = df_sales.groupby(df_sales['Order_Date'].dt.to_period('M')).agg({
    'Revenue': 'sum',
    'Order_ID': 'nunique',
    'Customer_ID': 'nunique',
    'Quantity': 'sum'
}).reset_index()

monthly_kpis['Order_Date'] = monthly_kpis['Order_Date'].astype(str)
monthly_kpis['AOV'] = monthly_kpis['Revenue'] / monthly_kpis['Order_ID']
monthly_kpis['Revenue_Per_Customer'] = monthly_kpis['Revenue'] / monthly_kpis['Customer_ID']

# Calculate month-over-month growth
monthly_kpis['Revenue_Growth'] = monthly_kpis['Revenue'].pct_change() * 100
monthly_kpis['Customer_Growth'] = monthly_kpis['Customer_ID'].pct_change() * 100

print("\nMonthly KPI Trends:")
print(monthly_kpis.tail(6))

# Visualize KPI trends
fig = go.Figure()

fig.add_trace(go.Scatter(x=monthly_kpis['Order_Date'], y=monthly_kpis['Revenue'],
                         mode='lines+markers', name='Revenue',
                         line=dict(color='blue', width=3)))

fig.add_trace(go.Scatter(x=monthly_kpis['Order_Date'], y=monthly_kpis['Order_ID']*100,
                         mode='lines+markers', name='Orders (x100)',
                         line=dict(color='green', width=2), yaxis='y2'))

fig.update_layout(
    title='Monthly Revenue & Orders Trend',
    xaxis=dict(title='Month'),
    yaxis=dict(title='Revenue ($)', side='left'),
    yaxis2=dict(title='Number of Orders', overlaying='y', side='right'),
    hovermode='x unified',
    height=500
)
fig.write_html('outputs/figures/monthly_kpi_trends.html')

# Save monthly KPIs
monthly_kpis.to_csv('data/processed/monthly_kpis.csv', index=False)
```

### **Step 3: Category-Level KPIs**

```python
# KPIs by product category
category_kpis = df_sales.groupby('Product_Category').agg({
    'Revenue': 'sum',
    'Order_ID': 'nunique',
    'Customer_ID': 'nunique',
    'Quantity': 'sum',
    'Product_ID': 'nunique'
}).reset_index()

category_kpis['AOV'] = category_kpis['Revenue'] / category_kpis['Order_ID']
category_kpis['Revenue_Share'] = (category_kpis['Revenue'] / category_kpis['Revenue'].sum()) * 100
category_kpis = category_kpis.sort_values('Revenue', ascending=False)

print("\nCategory-Level KPIs:")
print(category_kpis)

# Visualize category performance
fig = px.treemap(category_kpis, path=['Product_Category'], values='Revenue',
                 color='AOV',
                 title='Category Performance: Revenue & AOV',
                 color_continuous_scale='RdYlGn')
fig.write_html('outputs/figures/category_kpis_treemap.html')

# Save category KPIs
category_kpis.to_csv('outputs/reports/category_kpis.csv', index=False)
```

### **Step 4: Regional KPIs**

```python
# KPIs by region
regional_kpis = df_sales.groupby('Region').agg({
    'Revenue': 'sum',
    'Order_ID': 'nunique',
    'Customer_ID': 'nunique',
    'Quantity': 'sum'
}).reset_index()

regional_kpis['AOV'] = regional_kpis['Revenue'] / regional_kpis['Order_ID']
regional_kpis['Customer_Penetration'] = (regional_kpis['Customer_ID'] / kpis['Total_Customers']) * 100

print("\nRegional KPIs:")
print(regional_kpis)

# Save regional KPIs
regional_kpis.to_csv('outputs/reports/regional_kpis.csv', index=False)
```

---

## 11. DASHBOARD CREATION

### **Step 1: Power BI Dashboard (Recommended)**

**Instructions for Power BI:**

1. **Install Power BI Desktop** (Free from Microsoft)

2. **Import Data:**
   - Open Power BI Desktop
   - Click "Get Data" → "Text/CSV"
   - Import the following files:
     - `data/processed/cleaned_retail_sales.csv`
     - `data/processed/customer_segments.csv`
     - `data/processed/rfm_analysis.csv`
     - `data/processed/monthly_kpis.csv`

3. **Create Relationships:**
   - Go to "Model" view
   - Connect tables via `Customer_ID` and other common keys
   - Set appropriate cardinality (1-to-many, many-to-one)

4. **Build Dashboard Pages:**

   **Page 1: Executive Summary**
   - KPI Cards: Total Revenue, Total Customers, AOV, CLV
   - Line Chart: Monthly Revenue Trend
   - Bar Chart: Revenue by Category
   - Map Visual: Revenue by Region

   **Page 2: Customer Analytics**
   - Donut Chart: Customer Segments Distribution
   - Table: Top 10 Customers
   - Scatter Plot: RFM Analysis (Recency vs Frequency, size by Monetary)
   - Cohort Retention Heatmap

   **Page 3: Product Performance**
   - Treemap: Category Performance
   - Bar Chart: Top 20 Products by Revenue
   - Line Chart: Product Sales Trend
   - Table: Product Performance Metrics

   **Page 4: Marketing Insights**
   - Funnel Chart: Customer Journey
   - Gauge Charts: Retention Rate, Churn Rate
   - Column Chart: CLV by Segment
   - Matrix: Campaign Performance (if available)

5. **Add Interactivity:**
   - Add slicers for Date Range, Category, Region
   - Enable cross-filtering between visuals
   - Add drill-through pages for detailed analysis

6. **Format Dashboard:**
   - Apply consistent color scheme
   - Add company logo and title
   - Use appropriate fonts and sizing
   - Add tooltips for user guidance

7. **Publish:**
   - Save as `.pbix` file in `dashboards/` folder
   - Export as PDF for presentation
   - Publish to Power BI Service (if available)

### **Step 2: Tableau Dashboard (Alternative)**

**Instructions for Tableau:**

1. **Install Tableau Public** (Free version)

2. **Connect to Data:**
   - Open Tableau
   - Connect to CSV files from `data/processed/`
   - Join tables on common keys

3. **Create Worksheets:**
   - Revenue Trend
   - Customer Segments
   - Product Performance
   - Regional Analysis

4. **Build Dashboard:**
   - Drag worksheets onto dashboard canvas
   - Arrange in logical flow
   - Add filters and parameters
   - Enable dashboard actions

5. **Publish:**
   - Save workbook
   - Publish to Tableau Public
   - Share link in README

### **Step 3: Python Dash Dashboard (Interactive Web App)**

```python
# Install required packages
# pip install dash plotly dash-bootstrap-components

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

# Load processed data
df_sales = pd.read_csv('data/processed/cleaned_retail_sales.csv')
rfm = pd.read_csv('data/processed/customer_segments.csv')
monthly_kpis = pd.read_csv('data/processed/monthly_kpis.csv')

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Retail & Marketing Analytics Dashboard", 
                        className="text-center mb-4"), width=12)
    ]),
    
    # KPI Cards
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4(f"${kpis['Total_Revenue']:,.0f}", className="card-title"),
                html.P("Total Revenue", className="card-text")
            ])
        ]), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4(f"{kpis['Total_Customers']:,}", className="card-title"),
                html.P("Total Customers", className="card-text")
            ])
        ]), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4(f"${kpis['Avg_Order_Value']:,.2f}", className="card-title"),
                html.P("Avg Order Value", className="card-text")
            ])
        ]), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4(f"{kpis['CLV_to_CAC_Ratio']:.2f}x", className="card-title"),
                html.P("CLV/CAC Ratio", className="card-text")
            ])
        ]), width=3)
    ], className="mb-4"),
    
    # Charts Row 1
    dbc.Row([
        dbc.Col(dcc.Graph(id='revenue-trend'), width=8),
        dbc.Col(dcc.Graph(id='customer-segments'), width=4)
    ], className="mb-4"),
    
    # Charts Row 2
    dbc.Row([
        dbc.Col(dcc.Graph(id='category-performance'), width=6),
        dbc.Col(dcc.Graph(id='regional-sales'), width=6)
    ])
    
], fluid=True)

# Callbacks for interactivity
@app.callback(
    Output('revenue-trend', 'figure'),
    Input('revenue-trend', 'id')
)
def update_revenue_trend(_):
    fig = px.line(monthly_kpis, x='Order_Date', y='Revenue',
                  title='Monthly Revenue Trend',
                  markers=True)
    return fig

@app.callback(
    Output('customer-segments', 'figure'),
    Input('customer-segments', 'id')
)
def update_segments(_):
    segment_counts = rfm['Cluster_Name'].value_counts()
    fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                 title='Customer Segments',
                 hole=0.4)
    return fig

@app.callback(
    Output('category-performance', 'figure'),
    Input('category-performance', 'id')
)
def update_category(_):
    category_sales = df_sales.groupby('Product_Category')['Revenue'].sum().sort_values(ascending=False)
    fig = px.bar(x=category_sales.index, y=category_sales.values,
                 title='Revenue by Category',
                 labels={'x': 'Category', 'y': 'Revenue ($)'})
    return fig

@app.callback(
    Output('regional-sales', 'figure'),
    Input('regional-sales', 'id')
)
def update_regional(_):
    regional_sales = df_sales.groupby('Region')['Revenue'].sum()
    fig = px.pie(values=regional_sales.values, names=regional_sales.index,
                 title='Revenue by Region')
    return fig

# Run app
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
```

**Save as:** `dashboards/dash_app.py`

**To run:**
```bash
python dashboards/dash_app.py
# Open browser at http://127.0.0.1:8050
```

---

## 12. INSIGHTS & RECOMMENDATIONS

### **Key Findings:**

**1. Customer Segmentation Insights:**
- **VIP Customers** (Top 15%): Generate 45-50% of total revenue
  - High frequency (5+ orders/year), low recency (<30 days)
  - Recommendation: Implement VIP loyalty program with exclusive benefits

- **High Value Customers** (25%): Consistent purchasers with good spend
  - Recommendation: Targeted cross-sell/upsell campaigns

- **Potential Customers** (30%): Recent customers with growth potential
  - Recommendation: Nurture with personalized product recommendations

- **At-Risk Customers** (30%): High recency, declining engagement
  - Recommendation: Win-back campaigns with special offers

**2. Product Performance:**
- Top 20% of products generate 65% of revenue (Pareto Principle)
- Category X has highest AOV but lowest volume
- Seasonal patterns detected in categories Y and Z

**Recommendations:**
- Focus inventory on high-performing SKUs
- Bundle slow-moving items with bestsellers
- Optimize pricing strategy for high-AOV categories

**3. Temporal Patterns:**
- Peak sales months: November, December (holiday season)
- Weekend sales 25% higher than weekdays
- Quarter 4 generates 35% of annual revenue

**Recommendations:**
- Increase marketing spend in Q3 to prepare for Q4
- Weekend-specific promotions
- Build inventory ahead of peak season

**4. Regional Performance:**
- Region A: Highest revenue but declining growth
- Region C: Smallest market but fastest growing (+45% YoY)
- Significant variation in product preferences by region

**Recommendations:**
- Investigate decline in Region A
- Invest in marketing for Region C
- Localize product assortment by region

**5. Marketing Effectiveness:**
- CLV/CAC ratio of 3.2x indicates healthy customer economics
- Retention rate of 42% has room for improvement
- Repeat customers have 3x higher AOV

**Recommendations:**
- Implement retention-focused initiatives (loyalty program)
- Optimize acquisition channels with low CAC
- Create referral program leveraging existing customers

### **Action Plan:**

**Immediate Actions (Next 30 days):**
1. Launch VIP customer recognition program
2. Initiate win-back campaign for at-risk segment
3. Optimize top 20 product listings and inventory

**Short-term (2-3 months):**
1. Implement personalized email marketing by segment
2. Test dynamic pricing for different customer segments
3. Expand presence in high-growth regions

**Long-term (6-12 months):**
1. Build predictive churn model
2. Implement recommendation engine
3. Develop mobile app for enhanced engagement

---

## 13. GITHUB REPOSITORY SETUP

### **Step 1: Initialize Git Repository**

```bash
# Navigate to project directory
cd retail-marketing-analytics

# Initialize Git
git init

# Create .gitignore file
cat > .gitignore << EOF
# Data files
*.csv
*.xlsx
*.db
data/raw/*
!data/raw/.gitkeep

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv

# Jupyter Notebook
.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Outputs (keep structure, not files)
outputs/figures/*
outputs/reports/*
!outputs/figures/.gitkeep
!outputs/reports/.gitkeep

# Dashboard files (too large)
*.pbix
dashboards/*.twb
EOF

# Create .gitkeep files to preserve folder structure
touch data/raw/.gitkeep
touch outputs/figures/.gitkeep
touch outputs/reports/.gitkeep
```

### **Step 2: Create README.md**

```markdown
# 🛍️ Retail & Marketing Analytics Project

## Overview
End-to-end data analytics project focused on customer segmentation, sales optimization, and marketing effectiveness for retail businesses.

## 🎯 Objectives
- Perform customer segmentation using RFM analysis and K-Means clustering
- Analyze sales trends and product performance
- Design and track key performance indicators (KPIs)
- Build interactive dashboards for business insights
- Provide actionable recommendations for marketing strategy

## 📊 Dataset
- **Source:** Kaggle - Retail Sales Dataset
- **Records:** XX,XXX transactions
- **Time Period:** YYYY-YYYY
- **Features:** Customer ID, Product details, Sales, Dates, etc.

## 🛠️ Technologies Used
- **Python 3.9+**: pandas, numpy, scikit-learn, matplotlib, seaborn, plotly
- **Power BI**: Interactive dashboards
- **Jupyter Notebook**: Analysis and documentation
- **Git/GitHub**: Version control

## 📁 Project Structure
```
retail-marketing-analytics/
│
├── data/
│   ├── raw/                    # Original datasets
│   └── processed/              # Cleaned and transformed data
│
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_customer_segmentation.ipynb
│   └── 05_insights_recommendations.ipynb
│
├── scripts/
│   ├── data_processing.py
│   ├── clustering.py
│   └── kpi_calculation.py
│
├── dashboards/
│   ├── power_bi_dashboard.pbix
│   └── dash_app.py
│
├── outputs/
│   ├── figures/               # Visualizations
│   └── reports/               # Analysis reports
│
├── docs/
│   └── data_dictionary.csv
│
├── README.md
├── requirements.txt
└── .gitignore
```

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.9+
Jupyter Notebook
Power BI Desktop (optional)
```

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/retail-marketing-analytics.git
cd retail-marketing-analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis
```bash
# Run Jupyter notebooks in order
jupyter notebook

# Or run Python scripts
python scripts/data_processing.py
python scripts/clustering.py

# Launch Dash dashboard
python dashboards/dash_app.py
```

## 📈 Key Results

### Customer Segments Identified
1. **VIP Customers** (15%): High frequency, low recency, high spend
2. **High Value** (25%): Consistent purchasers
3. **Potential** (30%): Recent customers with growth potential
4. **At Risk** (30%): Declining engagement

### Performance Metrics
- **Total Revenue:** $X.XX Million
- **Customer Retention Rate:** XX%
- **Average CLV:** $X,XXX
- **CLV/CAC Ratio:** X.Xx

### Key Insights
- Top 20% products generate 65% of revenue
- Weekend sales 25% higher than weekdays
- Q4 generates 35% of annual revenue
- VIP segment contributes 45% of total revenue

## 📊 Dashboards
- **Power BI Dashboard:** Comprehensive interactive dashboard with 4 pages
- **Python Dash App:** Web-based analytics application
- **Tableau Dashboard:** Alternative visualization platform

## 🎓 Learning Outcomes
- Data cleaning and preprocessing techniques
- Exploratory data analysis (EDA)
- RFM analysis and customer segmentation
- K-Means clustering implementation
- KPI design and tracking
- Dashboard creation and storytelling
- Business insights generation

## 📝 Project Report
Full project documentation available in `docs/project_report.pdf`

## 👨‍💻 Author
**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

## 📄 License
This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments
- Dataset source: Kaggle
- Inspiration: Various retail analytics case studies
- Tools: Python, Power BI, scikit-learn

## 📧 Contact
For questions or collaboration opportunities, please reach out via email or LinkedIn.
```

**Save as:** `README.md`

### **Step 3: Create requirements.txt**

```text
# Data Processing
pandas==2.0.3
numpy==1.24.3
openpyxl==3.1.2

# Visualization
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0

# Machine Learning
scikit-learn==1.3.0
scipy==1.11.1

# Dashboard
dash==2.11.1
dash-bootstrap-components==1.4.2

# Utilities
jupyter==1.0.0
notebook==6.5.4
kaggle==1.5.16

# Market Basket Analysis
mlxtend==0.22.0

# Optional
python-dateutil==2.8.2
```

**Save as:** `requirements.txt`

### **Step 4: Create Project Documentation**

```python
# Create comprehensive project documentation
project_docs = """
# RETAIL & MARKETING ANALYTICS
## Technical Documentation

### Data Pipeline
1. Data Acquisition → Kaggle API
2. Data Cleaning → Handle missing values, outliers, duplicates
3. Feature Engineering → Time-based, customer, product metrics
4. Analysis → EDA, RFM, Clustering, Market Basket
5. Visualization → Dashboards and reports

### Analysis Methodology

#### RFM Analysis
- **Recency:** Days since last purchase
- **Frequency:** Total number of purchases
- **Monetary:** Total spending

Customers scored 1-5 on each dimension, then segmented based on combined scores.

#### K-Means Clustering
- Features: Recency, Frequency, Monetary (standardized)
- Optimal clusters: 4 (determined via elbow method and silhouette score)
- Segments: VIP, High Value, Potential, At Risk

#### Market Basket Analysis
- Algorithm: Apriori
- Minimum support: 0.01
- Minimum confidence: 0.3
- Metric: Lift > 1.0

### KPI Definitions

**Revenue Metrics:**
- Total Revenue: Sum of all sales
- AOV: Average revenue per order
- Revenue Per Customer: Total revenue / unique customers

**Customer Metrics:**
- Retention Rate: % customers with >1 purchase
- Churn Rate: % customers inactive >180 days
- CLV: Estimated lifetime value per customer

**Product Metrics:**
- Total SKUs: Number of unique products
- Avg Items Per Order: Mean quantity per transaction

### Statistical Tests Performed
- Correlation analysis for feature relationships
- ANOVA for category comparison
- Chi-square for categorical associations

### Assumptions & Limitations
1. Data is representative of overall business
2. Profit margin assumed at 25%
3. CAC estimated at $50
4. No data on marketing campaigns
5. Limited to transactional data only

### Future Enhancements
1. Predictive churn modeling
2. Real-time dashboard updates
3. A/B testing framework
4. Recommendation system
5. Time series forecasting
"""

with open('docs/technical_documentation.md', 'w') as f:
    f.write(project_docs)

print("Documentation created successfully!")
```

### **Step 5: Add and Commit to Git**

```bash
# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Complete retail & marketing analytics project"

# Create GitHub repository (via GitHub website)
# Then connect local to remote
git remote add origin https://github.com/yourusername/retail-marketing-analytics.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### **Step 6: Create GitHub Repository Description**

**Repository Description:**
```
End-to-end retail analytics project featuring customer segmentation (K-Means, RFM), 
sales analysis, KPI tracking, and interactive Power BI dashboards. Python | pandas | 
scikit-learn | Plotly | Power BI
```

**Topics/Tags:**
```
retail-analytics, customer-segmentation, data-science, rfm-analysis, 
kmeans-clustering, power-bi, python, pandas, data-visualization, 
marketing-analytics, kpi-dashboard, business-intelligence
```

---

## 14. EXECUTIVE SUMMARY

### **RETAIL & MARKETING ANALYTICS PROJECT**
### **Executive Summary Report**

---

#### **1. PROJECT OVERVIEW**

This comprehensive analytics project analyzed [XX,XXX] retail transactions to uncover customer behavior patterns, optimize marketing strategies, and improve business performance. Using advanced data science techniques including RFM analysis, K-Means clustering, and market basket analysis, we identified actionable insights to drive revenue growth and customer retention.

---

#### **2. KEY BUSINESS METRICS**

**Financial Performance:**
- **Total Revenue:** $XX.X Million
- **Average Order Value:** $XXX
- **Total Profit (Est.):** $X.X Million
- **Revenue Per Customer:** $X,XXX

**Customer Metrics:**
- **Total Customers:** XX,XXX
- **Customer Retention Rate:** XX%
- **Repeat Purchase Rate:** XX%
- **Customer Lifetime Value:** $X,XXX
- **CLV/CAC Ratio:** X.Xx (Healthy: >3.0)

**Operational Metrics:**
- **Total Orders:** XX,XXX
- **Total Products Sold:** XXX,XXX units
- **Average Items Per Order:** X.X
- **Total SKUs:** X,XXX

---

#### **3. CUSTOMER SEGMENTATION ANALYSIS**

We identified four distinct customer segments using K-Means clustering on RFM metrics:

| Segment | Size | Avg Recency | Avg Frequency | Avg Monetary | Revenue Contribution |
|---------|------|-------------|---------------|--------------|---------------------|
| **VIP Customers** | 15% | 25 days | 8.5 orders | $3,500 | 45% |
| **High Value** | 25% | 45 days | 5.2 orders | $1,800 | 30% |
| **Potential** | 30% | 60 days | 2.8 orders | $900 | 18% |
| **At Risk** | 30% | 180 days | 1.5 orders | $450 | 7% |

**Key Finding:** Just 15% of customers (VIP segment) generate 45% of total revenue, highlighting the importance of retention strategies for high-value customers.

---

#### **4. PRODUCT & CATEGORY INSIGHTS**

**Top Performing Categories:**
1. **Electronics:** $X.X Million (XX% of revenue)
2. **Home & Garden:** $X.X Million (XX% of revenue)
3. **Fashion:** $X.X Million (XX% of revenue)

**Key Findings:**
- **Pareto Principle Confirmed:** Top 20% of products generate 65% of revenue
- **Seasonal Patterns:** Categories show distinct seasonal trends with Q4 peak
- **Cross-Sell Opportunities:** Market basket analysis revealed 50+ high-lift product associations
- **Regional Variations:** Product preferences vary significantly across regions

---

#### **5. TEMPORAL PATTERNS**

**Monthly Trends:**
- Peak months: November (+35%), December (+40%)
- Lowest month: February (-20%)
- Clear seasonal pattern with Q4 surge

**Weekly Patterns:**
- Weekend sales 25% higher than weekdays
- Friday shows highest conversion rate
- Monday lowest performing day

**Implications:** Optimize marketing calendar and inventory planning around identified patterns.

---

#### **6. REGIONAL PERFORMANCE**

| Region | Revenue | Growth YoY | Market Share | AOV |
|--------|---------|------------|--------------|-----|
| Region A | $X.XM | -5% | 35% | $XXX |
| Region B | $X.XM | +12% | 30% | $XXX |
| Region C | $X.XM | +45% | 20% | $XXX |
| Region D | $X.XM | +8% | 15% | $XXX |

**Alert:** Region A (largest market) showing decline despite highest AOV - requires immediate attention.

---

#### **7. MARKETING EFFECTIVENESS**

**Customer Acquisition:**
- **CAC (Customer Acquisition Cost):** $50
- **Payback Period:** ~3 months
- **ROI:** Positive across all channels

**Customer Retention:**
- **Current Retention Rate:** 42%
- **Industry Benchmark:** 45-50%
- **Gap:** -3 to -8 percentage points

**Opportunity:** Improving retention by 5% could increase profits by 25-95% (industry research).

---

#### **8. STRATEGIC RECOMMENDATIONS**

**Immediate Actions (0-30 Days):**

1. **Launch VIP Loyalty Program**
   - Target: Top 15% customers
   - Benefits: Exclusive discounts, early access, personalized service
   - Expected Impact: +10% retention in VIP segment

2. **Win-Back Campaign**
   - Target: At-Risk customers (180+ days inactive)
   - Tactic: 20% discount offer + personalized email
   - Expected Impact: Reactivate 15% of churned customers

3. **Optimize Top 20 Products**
   - Action: Improve listings, increase inventory, featured placement
   - Expected Impact: +8% revenue from top products

**Short-Term Initiatives (2-3 Months):**

4. **Segmented Email Marketing**
   - Personalized campaigns by customer segment
   - Expected Impact: +15% email conversion rate

5. **Dynamic Pricing Strategy**
   - Test price optimization for different segments
   - Expected Impact: +5% overall margin

6. **Regional Expansion Focus**
   - Double down on Region C (fastest growing)
   - Expected Impact: +30% growth in target region

**Long-Term Strategy (6-12 Months):**

7. **Predictive Churn Model**
   - Machine learning model to identify at-risk customers early
   - Expected Impact: Reduce churn by 20%

8. **Product Recommendation Engine**
   - Personalized recommendations based on purchase history
   - Expected Impact: +12% cross-sell revenue

9. **Mobile App Development**
   - Enhanced customer engagement and loyalty
   - Expected Impact: +25% mobile channel revenue

---

#### **9. FINANCIAL IMPACT PROJECTION**

**Conservative Estimates (12-Month Horizon):**

| Initiative | Investment | Projected Return | ROI |
|------------|------------|------------------|-----|
| VIP Loyalty Program | $50K | $200K | 300% |
| Win-Back Campaign | $25K | $150K | 500% |
| Email Marketing | $30K | $180K | 500% |
| Churn Prevention | $75K | $300K | 300% |
| **TOTAL** | **$180K** | **$830K** | **361%** |

**Net Impact:** $650K additional profit in first year.

---

#### **10. RISK ASSESSMENT**

**Identified Risks:**
1. **Market Concentration:** 45% revenue from 15% customers (mitigation: diversify)
2. **Regional Decline:** Region A showing negative growth (action: investigate and address)
3. **High Churn:** 30% customers at risk (solution: retention initiatives)
4. **Seasonal Dependency:** 35% revenue in Q4 (strategy: even out demand)

**Mitigation Strategy:** Diversified approach across multiple recommendations to spread risk.

---

#### **11. NEXT STEPS**

**Week 1-2:**
- Present findings to executive team
- Prioritize recommendations based on impact/effort matrix
- Allocate budget for approved initiatives

**Week 3-4:**
- Launch VIP program pilot
- Deploy win-back campaign
- Begin A/B testing for email personalization

**Month 2-3:**
- Monitor KPIs and campaign performance
- Iterate based on early results
- Scale successful initiatives

**Ongoing:**
- Weekly KPI dashboard reviews
- Monthly deep-dive analytics
- Quarterly strategy refinement

---

#### **12. CONCLUSION**

This analysis reveals significant opportunities to optimize retail and marketing performance through data-driven customer segmentation, targeted campaigns, and strategic focus on high-value segments. 

**The path forward is clear:**
1. Protect and grow VIP customer relationships
2. Prevent churn through proactive engagement
3. Optimize product mix and inventory
4. Personalize marketing by segment
5. Expand in high-growth regions

**Expected Outcome:** By implementing these recommendations, we project a **15-20% increase in revenue** and **25-30% improvement in customer retention** within the first year, translating to **$650K+ in additional profit**.

---

**Prepared By:** [Your Name]  
**Date:** [Current Date]  
**Contact:** your.email@example.com

---

### **APPENDICES**

**Appendix A:** Detailed Cluster Profiles  
**Appendix B:** Market Basket Analysis Results  
**Appendix C:** Statistical Test Results  
**Appendix D:** Dashboard Screenshots  
**Appendix E:** Data Dictionary

---

*This executive summary is based on comprehensive analysis of retail transaction data using industry-standard analytics methodologies including RFM analysis, K-Means clustering, cohort analysis, and statistical testing.*

---

## 15. PRESENTATION SLIDES OVERVIEW

### **Slide Deck Structure (5-7 Slides)**

---

### **SLIDE 1: TITLE SLIDE**

**Title:** Retail & Marketing Analytics: Customer Segmentation & Growth Strategy

**Subtitle:** Data-Driven Insights for Revenue Optimization

**Content:**
- Project Name
- Your Name
- Date
- Course/Organization

**Visual:** Professional background with retail imagery or data visualization theme

---

### **SLIDE 2: BUSINESS PROBLEM & OBJECTIVES**

**Title:** Business Challenge & Project Goals

**Left Column - The Problem:**
- Declining customer retention (42% vs 50% benchmark)
- Limited understanding of customer behavior
- Inefficient marketing spend
- Inconsistent sales across regions

**Right Column - Our Objectives:**
- ✓ Identify high-value customer segments
- ✓ Analyze sales patterns and trends
- ✓ Design actionable KPIs
- ✓ Build interactive dashboards
- ✓ Provide strategic recommendations

**Visual:** Split layout with icons for problems and checkmarks for objectives

---

### **SLIDE 3: DATA & METHODOLOGY**

**Title:** Analytical Approach & Techniques

**Dataset Overview:**
- **Source:** Kaggle Retail Sales Data
- **Size:** XX,XXX transactions
- **Period:** YYYY-YYYY
- **Features:** Customers, Products, Sales, Dates, Regions

**Methodology:**
```
Data Acquisition → Cleaning → EDA → Advanced Analytics → Insights
```

**Techniques Applied:**
1. **RFM Analysis** - Customer behavior segmentation
2. **K-Means Clustering** - 4 distinct customer groups
3. **Market Basket Analysis** - Product associations
4. **Cohort Analysis** - Retention tracking
5. **KPI Design** - 15+ metrics tracked

**Visual:** Process flow diagram + icons for each technique

---

### **SLIDE 4: CUSTOMER SEGMENTATION INSIGHTS**

**Title:** Four Distinct Customer Segments Identified

**Visual:** Large pie chart showing segment distribution with callout boxes

**Segment Breakdown:**

**🏆 VIP Customers (15%)**
- Contribute 45% of revenue
- Recent, frequent, high-spending
- Action: Premium loyalty program

**💎 High Value (25%)**
- Consistent purchasers
- 30% revenue contribution
- Action: Cross-sell campaigns

**🌱 Potential (30%)**
- Recent customers, growth opportunity
- 18% revenue contribution
- Action: Nurture with offers

**⚠️ At Risk (30%)**
- Declining engagement
- 7% revenue contribution
- Action: Win-back campaigns

**Key Insight Box:** "Just 15% of customers generate 45% of revenue - focus retention efforts here!"

---

### **SLIDE 5: KEY FINDINGS & PERFORMANCE METRICS**

**Title:** Performance Highlights & Trends

**Top Half - KPI Dashboard Style:**
- **Total Revenue:** $XX.X Million
- **Customer Retention:** XX%
- **CLV/CAC Ratio:** X.Xx
- **Avg Order Value:** $XXX

**Bottom Half - Key Insights:**

**📊 Product Performance**
- Top 20% products = 65% revenue (Pareto confirmed)
- Electronics category leads with XX% share

**📅 Temporal Patterns**
- Q4 drives 35% of annual revenue
- Weekend sales +25% vs weekdays

**🌍 Regional Analysis**
- Region C: Fastest growing (+45% YoY)
- Region A: Declining despite largest share (alert!)

**Visual:** Mix of KPI cards and mini charts/graphs

---

### **SLIDE 6: STRATEGIC RECOMMENDATIONS**

**Title:** Action Plan for Growth

**3-Column Layout:**

**🚀 IMMEDIATE (0-30 Days)**
- Launch VIP Loyalty Program
  - Target: Top 15%
  - Impact: +10% retention
- Win-Back Campaign
  - Target: At-Risk segment
  - Impact: Reactivate 15%
- Optimize Top 20 Products
  - Impact: +8% revenue

**📈 SHORT-TERM (2-3 Months)**
- Segmented Email Marketing
  - Impact: +15% conversion
- Dynamic Pricing Tests
  - Impact: +5% margin
- Regional Expansion (Region C)
  - Impact: +30% growth

**🎯 LONG-TERM (6-12 Months)**
- Predictive Churn Model
  - Impact: -20% churn
- Recommendation Engine
  - Impact: +12% cross-sell
- Mobile App Launch
  - Impact: +25% mobile revenue

**Visual:** Timeline or roadmap visual

---

### **SLIDE 7: BUSINESS IMPACT & NEXT STEPS**

**Title:** Projected Impact & Implementation Plan

**Left Side - Financial Projection:**

**12-Month ROI Forecast:**
- Total Investment: $180K
- Projected Returns: $830K
- **Net Profit Impact: $650K**
- **Overall ROI: 361%**

**Expected Outcomes:**
- ✓ 15-20% revenue increase
- ✓ 25-30% retention improvement
- ✓ 10%+ customer lifetime value growth

**Right Side - Next Steps:**

**Week 1-2:** Executive approval & budget allocation  
**Week 3-4:** Launch pilot programs  
**Month 2-3:** Monitor & optimize  
**Ongoing:** Dashboard tracking & iteration

**Bottom Banner:**
"Data-driven decisions lead to measurable results. Let's transform insights into action!"

**Visual:** Growth arrow chart + checklist

---

### **OPTIONAL SLIDE 8: DASHBOARD DEMO**

**Title:** Interactive Analytics Dashboard

**Content:**
- Screenshot of Power BI/Tableau dashboard
- Highlight key features:
  - Real-time KPI tracking
  - Drill-down capabilities
  - Segment filtering
  - Trend analysis

**Call-to-Action:** "Live demo available - let's explore the data together!"

---

### **Presentation Tips:**

1. **Keep text minimal** - Use bullet points and visuals
2. **Tell a story** - Problem → Analysis → Insights → Action
3. **Use consistent colors** - Brand colors or professional palette
4. **Include data visualizations** - Charts more than text
5. **Practice timing** - 7-10 minutes for full presentation
6. **Prepare for questions** - Know your data deeply
7. **Have backup slides** - Technical details, methodology deep-dives

---

### **Design Recommendations:**

**Color Scheme:**
- Primary: Deep Blue (#1f4788)
- Secondary: Orange (#ff6b35)
- Accent: Green (#4caf50) for positive metrics
- Alert: Red (#f44336) for risks

**Fonts:**
- Headers: Montserrat Bold
- Body: Open Sans Regular
- Numbers: Roboto Medium

**Layout:**
- Consistent margins
- White space for readability
- Icons from Font Awesome or similar
- Professional charts (no 3D effects)

---

## 🎓 LEARNING CHECKLIST FOR STUDENTS

### **Skills Developed:**
- [ ] Data acquisition from Kaggle
- [ ] Data cleaning and preprocessing
- [ ] Exploratory data analysis (EDA)
- [ ] Statistical analysis
- [ ] RFM analysis
- [ ] K-Means clustering
- [ ] Market basket analysis
- [ ] KPI design and tracking
- [ ] Data visualization
- [ ] Dashboard creation (Power BI/Tableau)
- [ ] Business insight generation
- [ ] Technical documentation
- [ ] GitHub version control
- [ ] Presentation skills

### **Deliverables Checklist:**
- [ ] Cleaned dataset
- [ ] Jupyter notebooks (5+)
- [ ] Python scripts
- [ ] Interactive dashboard
- [ ] Visualizations (15+)
- [ ] Analysis reports
- [ ] Executive summary
- [ ] Presentation slides
- [ ] GitHub repository
- [ ] README documentation

---

## 📚 ADDITIONAL RESOURCES

### **Recommended Reading:**
1. "Data Science for Business" by Foster Provost
2. "Storytelling with Data" by Cole Nussbaumer Knaflic
3. "Marketing Analytics" by Wayne Winston

### **Online Courses:**
- Google Analytics Academy
- Coursera: Customer Analytics
- DataCamp: Marketing Analytics track

### **Useful Links:**
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Power BI Documentation](https://docs.microsoft.com/power-bi/)
- [scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [Plotly Documentation](https://plotly.com/python/)

---

## ✅ PROJECT COMPLETION CHECKLIST

### **Phase 1: Setup ✓**
- [ ] Project structure created
- [ ] Virtual environment setup
- [ ] Dependencies installed
- [ ] Data downloaded

### **Phase 2: Data Processing ✓**
- [ ] Data loaded and inspected
- [ ] Missing values handled
- [ ] Outliers treated
- [ ] Features engineered
- [ ] Cleaned data saved

### **Phase 3: Analysis ✓**
- [ ] EDA completed
- [ ] RFM analysis done
- [ ] Clustering performed
- [ ] KPIs calculated
- [ ] Market basket analysis

### **Phase 4: Visualization ✓**
- [ ] Charts created
- [ ] Dashboard built
- [ ] Reports generated

### **Phase 5: Documentation ✓**
- [ ] README written
- [ ] Technical docs created
- [ ] Executive summary prepared
- [ ] Presentation slides made

### **Phase 6: Deployment ✓**
- [ ] GitHub repository created
- [ ] Code pushed
- [ ] Documentation uploaded
- [ ] Project shared

---

## 🎯 SUCCESS CRITERIA

**Project is considered successful if:**
- ✓ All data quality issues addressed
- ✓ Minimum 4 customer segments identified
- ✓ 15+ KPIs tracked and visualized
- ✓ Interactive dashboard created
- ✓ Actionable recommendations provided
- ✓ Professional documentation completed
- ✓ GitHub repository well-organized
- ✓ Presentation ready for stakeholders

---

## 📞 SUPPORT & QUESTIONS

**For Students:**
- Review notebooks in sequential order
- Check documentation for clarifications
- Use GitHub Issues for questions
- Join study groups for collaboration

**For Instructors:**
- This project can be adapted for different datasets
- Adjust complexity based on student level
- Encourage creativity in visualizations
- Focus on business storytelling

---

**END OF PROJECT GUIDE**

*This comprehensive guide provides everything needed to complete a professional-level retail & marketing analytics project from start to finish. Follow each section carefully, experiment with the data, and most importantly - tell a compelling story with your insights!*

**Good luck with your project! 🚀📊**