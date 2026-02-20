# Retail & Marketing Analytics Project
# Notebook 2: Data Cleaning and Preprocessing

"""
Project: Retail & Marketing Analytics - Customer Segmentation & Sales Optimization
Notebook: 02 - Data Cleaning and Preprocessing
Author: [Your Name]
Date: [Current Date]

Objective:
- Handle missing values
- Remove duplicates
- Convert data types
- Detect and treat outliers
- Feature engineering
- Save cleaned dataset
"""

# ============================================================================
# 1. IMPORT LIBRARIES AND LOAD DATA
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Load the raw data
df_sales = pd.read_csv('data/raw/retail_sales_data.csv')

print("="*80)
print("DATA CLEANING AND PREPROCESSING")
print("="*80)
print(f"\nInitial Dataset Shape: {df_sales.shape}")
print(f"Initial Memory Usage: {df_sales.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Create a copy for cleaning
df_clean = df_sales.copy()

# ============================================================================
# 2. HANDLE MISSING VALUES
# ============================================================================

print("\n" + "="*80)
print("STEP 1: HANDLING MISSING VALUES")
print("="*80)

# Check missing values
print("\n📊 Missing Values Before Treatment:")
missing_before = df_clean.isnull().sum()
missing_before_pct = (missing_before / len(df_clean) * 100).round(2)
missing_df = pd.DataFrame({
    'Column': missing_before.index,
    'Missing_Count': missing_before.values,
    'Missing_Percentage': missing_before_pct.values
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
print(missing_df)

# Strategy for handling missing values
print("\n🔧 Treatment Strategy:")

# Separate columns by type
numerical_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()

# Handle numerical missing values - Fill with median
for col in numerical_cols:
    if df_clean[col].isnull().sum() > 0:
        median_value = df_clean[col].median()
        df_clean[col].fillna(median_value, inplace=True)
        print(f"  ✓ {col}: Filled with median ({median_value:.2f})")

# Handle categorical missing values - Fill with mode or 'Unknown'
for col in categorical_cols:
    if df_clean[col].isnull().sum() > 0:
        if df_clean[col].mode().empty:
            df_clean[col].fillna('Unknown', inplace=True)
            print(f"  ✓ {col}: Filled with 'Unknown'")
        else:
            mode_value = df_clean[col].mode()[0]
            df_clean[col].fillna(mode_value, inplace=True)
            print(f"  ✓ {col}: Filled with mode ('{mode_value}')")

# Verify no missing values remain
print(f"\n✅ Missing values after treatment: {df_clean.isnull().sum().sum()}")

# ============================================================================
# 3. REMOVE DUPLICATES
# ============================================================================

print("\n" + "="*80)
print("STEP 2: REMOVING DUPLICATES")
print("="*80)

# Check duplicates
duplicates_before = df_clean.duplicated().sum()
print(f"\n🔍 Duplicate rows found: {duplicates_before}")

# Remove duplicates
df_clean = df_clean.drop_duplicates()
print(f"✓ Duplicates removed: {duplicates_before}")
print(f"✓ New dataset shape: {df_clean.shape}")

# Reset index after removing duplicates
df_clean.reset_index(drop=True, inplace=True)

# ============================================================================
# 4. DATA TYPE CONVERSION
# ============================================================================

print("\n" + "="*80)
print("STEP 3: DATA TYPE CONVERSION")
print("="*80)

# Convert date columns to datetime
date_columns = ['Order_Date', 'Ship_Date']
for col in date_columns:
    if col in df_clean.columns:
        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        print(f"✓ Converted {col} to datetime")

# Convert categorical columns to category type for memory optimization
categorical_columns = ['Segment', 'Region', 'Product_Category', 'Product_Sub_Category', 'Order_Priority']
for col in categorical_columns:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].astype('category')
        print(f"✓ Converted {col} to category")

# Ensure numerical columns are correct type
if 'Quantity' in df_clean.columns:
    df_clean['Quantity'] = df_clean['Quantity'].astype('int32')
    
print(f"\n✅ Data types optimized!")
print(f"Memory usage after optimization: {df_clean.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================================================
# 5. OUTLIER DETECTION AND TREATMENT
# ============================================================================

print("\n" + "="*80)
print("STEP 4: OUTLIER DETECTION AND TREATMENT")
print("="*80)

def detect_outliers_iqr(data, column):
    """Detect outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Check for outliers in key numerical columns
outlier_columns = ['Sales', 'Quantity', 'Profit'] if 'Profit' in df_clean.columns else ['Sales', 'Quantity']

print("\n📊 Outlier Analysis:")
outlier_summary = []

for col in outlier_columns:
    if col in df_clean.columns:
        outliers, lb, ub = detect_outliers_iqr(df_clean, col)
        outlier_pct = (len(outliers) / len(df_clean) * 100)
        outlier_summary.append({
            'Column': col,
            'Outliers': len(outliers),
            'Percentage': f"{outlier_pct:.2f}%",
            'Lower_Bound': f"{lb:.2f}",
            'Upper_Bound': f"{ub:.2f}"
        })
        print(f"\n{col}:")
        print(f"  - Outliers detected: {len(outliers)} ({outlier_pct:.2f}%)")
        print(f"  - Lower bound: {lb:.2f}")
        print(f"  - Upper bound: {ub:.2f}")

# Visualize outliers before treatment
fig, axes = plt.subplots(1, len(outlier_columns), figsize=(15, 5))
if len(outlier_columns) == 1:
    axes = [axes]

for idx, col in enumerate(outlier_columns):
    if col in df_clean.columns:
        axes[idx].boxplot(df_clean[col].dropna())
        axes[idx].set_title(f'{col} Distribution\n(Before Treatment)', fontweight='bold')
        axes[idx].set_ylabel(col)
        axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/figures/03_outliers_before_treatment.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 03_outliers_before_treatment.png")
plt.show()

# Treatment: Cap outliers (Winsorization)
print("\n🔧 Outlier Treatment (Capping):")
for col in outlier_columns:
    if col in df_clean.columns:
        outliers, lb, ub = detect_outliers_iqr(df_clean, col)
        if len(outliers) > 0:
            # Cap values
            df_clean[f'{col}_Original'] = df_clean[col]  # Keep original for reference
            df_clean[col] = df_clean[col].clip(lower=lb, upper=ub)
            print(f"  ✓ {col}: Outliers capped to [{lb:.2f}, {ub:.2f}]")

# Visualize after treatment
fig, axes = plt.subplots(1, len(outlier_columns), figsize=(15, 5))
if len(outlier_columns) == 1:
    axes = [axes]

for idx, col in enumerate(outlier_columns):
    if col in df_clean.columns:
        axes[idx].boxplot(df_clean[col].dropna())
        axes[idx].set_title(f'{col} Distribution\n(After Treatment)', fontweight='bold')
        axes[idx].set_ylabel(col)
        axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/figures/04_outliers_after_treatment.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_outliers_after_treatment.png")
plt.show()

# ============================================================================
# 6. FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*80)
print("STEP 5: FEATURE ENGINEERING")
print("="*80)

# 1. Time-based features
if 'Order_Date' in df_clean.columns:
    print("\n⏰ Creating Time-Based Features:")
    df_clean['Year'] = df_clean['Order_Date'].dt.year
    df_clean['Month'] = df_clean['Order_Date'].dt.month
    df_clean['Month_Name'] = df_clean['Order_Date'].dt.month_name()
    df_clean['Quarter'] = df_clean['Order_Date'].dt.quarter
    df_clean['Day'] = df_clean['Order_Date'].dt.day
    df_clean['Day_of_Week'] = df_clean['Order_Date'].dt.dayofweek
    df_clean['Day_Name'] = df_clean['Order_Date'].dt.day_name()
    df_clean['Week_of_Year'] = df_clean['Order_Date'].dt.isocalendar().week
    df_clean['Is_Weekend'] = df_clean['Day_of_Week'].isin([5, 6]).astype(int)
    df_clean['Is_Month_Start'] = df_clean['Order_Date'].dt.is_month_start.astype(int)
    df_clean['Is_Month_End'] = df_clean['Order_Date'].dt.is_month_end.astype(int)
    print("  ✓ Year, Month, Quarter, Day, Day of Week")
    print("  ✓ Weekend indicator, Month start/end flags")

# 2. Revenue and profit metrics
if 'Sales' in df_clean.columns and 'Quantity' in df_clean.columns:
    print("\n💰 Creating Revenue Metrics:")
    df_clean['Unit_Price'] = df_clean['Sales'] / df_clean['Quantity']
    df_clean['Revenue'] = df_clean['Sales']
    
    if 'Discount' in df_clean.columns:
        df_clean['Discount_Amount'] = df_clean['Sales'] * df_clean['Discount']
        df_clean['Net_Revenue'] = df_clean['Sales'] - df_clean['Discount_Amount']
        print("  ✓ Unit Price, Revenue, Discount Amount, Net Revenue")
    
    if 'Profit' in df_clean.columns:
        df_clean['Profit_Margin'] = (df_clean['Profit'] / df_clean['Sales']) * 100
        df_clean['Profit_Ratio'] = df_clean['Profit'] / df_clean['Sales']
        print("  ✓ Profit Margin, Profit Ratio")

# 3. Delivery time calculation
if 'Order_Date' in df_clean.columns and 'Ship_Date' in df_clean.columns:
    print("\n🚚 Creating Delivery Metrics:")
    df_clean['Delivery_Days'] = (df_clean['Ship_Date'] - df_clean['Order_Date']).dt.days
    df_clean['Delivery_Category'] = pd.cut(df_clean['Delivery_Days'], 
                                           bins=[-np.inf, 2, 5, 10, np.inf],
                                           labels=['Same/Next Day', 'Standard', 'Delayed', 'Very Delayed'])
    print("  ✓ Delivery Days, Delivery Category")

# 4. Customer frequency (aggregated later, but flag here)
if 'Customer_ID' in df_clean.columns:
    print("\n👥 Creating Customer Flags:")
    customer_order_count = df_clean.groupby('Customer_ID').size()
    df_clean['Customer_Order_Count'] = df_clean['Customer_ID'].map(customer_order_count)
    df_clean['Is_Repeat_Customer'] = (df_clean['Customer_Order_Count'] > 1).astype(int)
    print("  ✓ Customer Order Count, Repeat Customer Flag")

# 5. Product performance metrics
if 'Product_ID' in df_clean.columns:
    print("\n📦 Creating Product Metrics:")
    product_sales = df_clean.groupby('Product_ID')['Sales'].agg(['sum', 'mean', 'count'])
    product_sales.columns = ['Product_Total_Sales', 'Product_Avg_Sales', 'Product_Order_Count']
    df_clean = df_clean.merge(product_sales, left_on='Product_ID', right_index=True, how='left')
    print("  ✓ Product Total Sales, Average Sales, Order Count")

# 6. Sales categories
if 'Sales' in df_clean.columns:
    print("\n📊 Creating Sales Categories:")
    df_clean['Sales_Category'] = pd.qcut(df_clean['Sales'], 
                                         q=4, 
                                         labels=['Low', 'Medium', 'High', 'Very High'],
                                         duplicates='drop')
    print("  ✓ Sales Category (Quartile-based)")

# 7. Season classification
if 'Month' in df_clean.columns:
    print("\n🌍 Creating Seasonal Features:")
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    }
    df_clean['Season'] = df_clean['Month'].map(season_map)
    print("  ✓ Season")

print(f"\n✅ Feature Engineering Completed!")
print(f"New dataset shape: {df_clean.shape}")

# ============================================================================
# 7. DATA VALIDATION
# ============================================================================

print("\n" + "="*80)
print("STEP 6: DATA VALIDATION")
print("="*80)

# Check for negative values in key columns
print("\n🔍 Checking for Invalid Values:")

validation_checks = []

if 'Sales' in df_clean.columns:
    negative_sales = (df_clean['Sales'] < 0).sum()
    validation_checks.append(f"Negative Sales: {negative_sales}")
    
if 'Quantity' in df_clean.columns:
    negative_qty = (df_clean['Quantity'] <= 0).sum()
    validation_checks.append(f"Zero/Negative Quantity: {negative_qty}")
    
if 'Discount' in df_clean.columns:
    invalid_discount = ((df_clean['Discount'] < 0) | (df_clean['Discount'] > 1)).sum()
    validation_checks.append(f"Invalid Discount: {invalid_discount}")

for check in validation_checks:
    print(f"  • {check}")

# Check date consistency
if 'Order_Date' in df_clean.columns and 'Ship_Date' in df_clean.columns:
    invalid_dates = (df_clean['Ship_Date'] < df_clean['Order_Date']).sum()
    print(f"  • Ship Date before Order Date: {invalid_dates}")

print("\n✅ Validation Complete!")

# ============================================================================
# 8. SUMMARY STATISTICS AFTER CLEANING
# ============================================================================

print("\n" + "="*80)
print("CLEANED DATA SUMMARY")
print("="*80)

print(f"\n📊 Final Dataset Shape: {df_clean.shape}")
print(f"📉 Records removed: {df_sales.shape[0] - df_clean.shape[0]}")
print(f"📈 Features added: {df_clean.shape[1] - df_sales.shape[1]}")
print(f"💾 Memory usage: {df_clean.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n📋 Data Quality Metrics:")
print(f"  • Missing values: {df_clean.isnull().sum().sum()}")
print(f"  • Duplicate rows: {df_clean.duplicated().sum()}")
print(f"  • Complete rows: {df_clean.dropna().shape[0]:,} ({(df_clean.dropna().shape[0]/len(df_clean)*100):.2f}%)")

print("\n📈 Key Statistics:")
if 'Sales' in df_clean.columns:
    print(f"  • Total Sales: ${df_clean['Sales'].sum():,.2f}")
    print(f"  • Average Order Value: ${df_clean['Sales'].mean():,.2f}")
if 'Customer_ID' in df_clean.columns:
    print(f"  • Unique Customers: {df_clean['Customer_ID'].nunique():,}")
if 'Product_ID' in df_clean.columns:
    print(f"  • Unique Products: {df_clean['Product_ID'].nunique():,}")

# ============================================================================
# 9. SAVE CLEANED DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 7: SAVING CLEANED DATA")
print("="*80)

# Save cleaned dataset
output_path = 'data/processed/cleaned_retail_sales.csv'
df_clean.to_csv(output_path, index=False)
print(f"✓ Cleaned data saved: {output_path}")

# Save data dictionary
data_dict = pd.DataFrame({
    'Column_Name': df_clean.columns,
    'Data_Type': df_clean.dtypes.values,
    'Non_Null_Count': df_clean.count().values,
    'Null_Count': df_clean.isnull().sum().values,
    'Unique_Values': [df_clean[col].nunique() for col in df_clean.columns],
    'Sample_Value': [df_clean[col].iloc[0] if len(df_clean) > 0 else None for col in df_clean.columns]
})

dict_path = 'docs/data_dictionary.csv'
data_dict.to_csv(dict_path, index=False)
print(f"✓ Data dictionary saved: {dict_path}")

# Create cleaning summary report
cleaning_report = f"""
DATA CLEANING AND PREPROCESSING REPORT
{'='*80}

1. DATA QUALITY IMPROVEMENTS
   Before Cleaning:
   - Records: {df_sales.shape[0]:,}
   - Missing Values: {df_sales.isnull().sum().sum():,}
   - Duplicates: {duplicates_before}
   
   After Cleaning:
   - Records: {df_clean.shape[0]:,}
   - Missing Values: {df_clean.isnull().sum().sum()}
   - Duplicates: {df_clean.duplicated().sum()}
   
   Records Removed: {df_sales.shape[0] - df_clean.shape[0]:,}
   Data Quality Score: {((1 - df_clean.isnull().sum().sum()/(df_clean.shape[0]*df_clean.shape[1])) * 100):.2f}%

2. FEATURES ENGINEERED
   Original Features: {df_sales.shape[1]}
   New Features: {df_clean.shape[1] - df_sales.shape[1]}
   Total Features: {df_clean.shape[1]}
   
   Categories of New Features:
   - Time-based: Year, Month, Quarter, Day, Week, Weekend
   - Revenue: Unit Price, Net Revenue, Discount Amount
   - Customer: Order Count, Repeat Customer Flag
   - Product: Total Sales, Average Sales, Order Count
   - Delivery: Delivery Days, Delivery Category
   - Categorical: Sales Category, Season

3. DATA TYPE OPTIMIZATION
   Memory Before: {df_sales.memory_usage(deep=True).sum() / 1024**2:.2f} MB
   Memory After: {df_clean.memory_usage(deep=True).sum() / 1024**2:.2f} MB
   Memory Saved: {(df_sales.memory_usage(deep=True).sum() - df_clean.memory_usage(deep=True).sum()) / 1024**2:.2f} MB

4. OUTLIER TREATMENT
   Method: IQR (Interquartile Range) with Winsorization
   Columns Treated: {', '.join(outlier_columns)}

5. NEXT STEPS
   ✓ Proceed to Exploratory Data Analysis (Notebook 03)
   - Univariate analysis
   - Bivariate analysis
   - Time series analysis
   - Customer behavior analysis

Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""

report_path = 'outputs/reports/02_cleaning_report.txt'
with open(report_path, 'w') as f:
    f.write(cleaning_report)
print(f"✓ Cleaning report saved: {report_path}")

print(cleaning_report)

print("\n" + "="*80)
print("NOTEBOOK 02 COMPLETED SUCCESSFULLY!")
print("="*80)
print("\n✅ Next Step: Open Notebook 03 - Exploratory Data Analysis")
print("="*80)
