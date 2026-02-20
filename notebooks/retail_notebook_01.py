# Retail & Marketing Analytics Project
# Notebook 1: Data Acquisition and Setup

"""
Project: Retail & Marketing Analytics - Customer Segmentation & Sales Optimization
Notebook: 01 - Data Acquisition and Setup
Author: [Your Name]
Date: [Current Date]

Objective:
- Setup project environment
- Download dataset from Kaggle
- Initial data inspection
- Create project folder structure
"""

# ============================================================================
# 1. IMPORT LIBRARIES
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from pathlib import Path
import zipfile

# Visualization settings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

print("Libraries imported successfully!")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")

# ============================================================================
# 2. CREATE PROJECT FOLDER STRUCTURE
# ============================================================================

# Define folder structure
folders = [
    'data/raw',
    'data/processed',
    'notebooks',
    'scripts',
    'dashboards',
    'outputs/figures',
    'outputs/reports',
    'docs'
]

# Create folders
for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)
    print(f"✓ Created: {folder}")

print("\nProject structure created successfully!")

# ============================================================================
# 3. KAGGLE HUB (OPTIONAL)
# ============================================================================
# Dataset: 
# https://www.kaggle.com/datasets/abdullah0a/retail-sales-data-with-seasonal-trends-and-marketing 

import kagglehub
import os
import shutil

# Step 1: Download dataset
path = kagglehub.dataset_download("abdullah0a/retail-sales-data-with-seasonal-trends-and-marketing")
print("Downloaded dataset path:", path)

# Step 2: Define new target directory and filename
target_dir = "data/raw"
os.makedirs(target_dir, exist_ok=True)  # create folder if not exist

# (Optional) If you want to rename the main CSV file
new_filename = "retail_sales_data.csv"  # rename as desired

# Step 3: Find the CSV file in downloaded folder
for file in os.listdir(path):
    if file.endswith(".csv"):
        source_file = os.path.join(path, file)
        destination_file = os.path.join(target_dir, new_filename)
        
        # Copy file (use shutil.move if you prefer moving instead of copying)
        shutil.copy(source_file, destination_file)
        print(f"Copied file to: {destination_file}")
        break
else:
    print("No CSV file found in the downloaded dataset folder.")

print("Place CSV file in:", target_dir)

# ============================================================================
# 4. LOAD DATASET
# ============================================================================

# Option 1: Load from Kaggle download
try:
    df_sales = pd.read_csv('data/raw/retail_sales_data.csv')
    print("\n✓ Dataset loaded successfully from Kaggle!")
except FileNotFoundError:
    print("\n⚠️ Dataset not found. Creating sample data for demonstration...")
    
    # Create sample dataset for demonstration
    np.random.seed(42)
    n_records = 10000
    
    # Sample data generation
    df_sales = pd.DataFrame({
        'Order_ID': [f'ORD{str(i).zfill(6)}' for i in range(1, n_records+1)],
        'Order_Date': pd.date_range(start='2022-01-01', periods=n_records, freq='H'),
        'Ship_Date': pd.date_range(start='2022-01-02', periods=n_records, freq='H'),
        'Customer_ID': [f'CUST{np.random.randint(1, 2000):04d}' for _ in range(n_records)],
        'Customer_Name': [f'Customer_{i}' for i in np.random.randint(1, 2000, n_records)],
        'Segment': np.random.choice(['Consumer', 'Corporate', 'Home Office'], n_records, p=[0.5, 0.3, 0.2]),
        'Region': np.random.choice(['East', 'West', 'Central', 'South'], n_records, p=[0.3, 0.25, 0.25, 0.2]),
        'Product_ID': [f'PROD{np.random.randint(1, 500):04d}' for _ in range(n_records)],
        'Product_Category': np.random.choice(['Electronics', 'Furniture', 'Office Supplies', 'Clothing'], 
                                            n_records, p=[0.3, 0.2, 0.3, 0.2]),
        'Product_Sub_Category': np.random.choice(['Phones', 'Chairs', 'Paper', 'Accessories', 'Tables', 
                                                  'Binders', 'Appliances', 'Shirts'], n_records),
        'Product_Name': [f'Product_{i}' for i in np.random.randint(1, 500, n_records)],
        'Sales': np.random.gamma(2, 50, n_records) + 10,
        'Quantity': np.random.randint(1, 10, n_records),
        'Discount': np.random.choice([0, 0.1, 0.15, 0.2, 0.25], n_records, p=[0.4, 0.2, 0.2, 0.15, 0.05]),
        'Profit': np.random.normal(20, 30, n_records),
        'Shipping_Cost': np.random.gamma(1.5, 3, n_records) + 2,
        'Order_Priority': np.random.choice(['Low', 'Medium', 'High', 'Critical'], n_records, p=[0.3, 0.4, 0.2, 0.1])
    })
    
    # Add some calculated fields
    df_sales['Unit_Price'] = df_sales['Sales'] / df_sales['Quantity']
    df_sales['Revenue'] = df_sales['Sales']
    
    # Add some missing values for realistic scenario
    df_sales.loc[np.random.choice(df_sales.index, 50, replace=False), 'Customer_Name'] = np.nan
    df_sales.loc[np.random.choice(df_sales.index, 30, replace=False), 'Profit'] = np.nan
    
    # Save sample data
    df_sales.to_csv('data/raw/retail_sales_data.csv', index=False)
    print("✓ Sample dataset created and saved!")

# ============================================================================
# 5. INITIAL DATA INSPECTION
# ============================================================================

print("\n" + "="*80)
print("INITIAL DATA INSPECTION")
print("="*80)

# Dataset shape
print(f"\n📊 Dataset Shape: {df_sales.shape[0]:,} rows × {df_sales.shape[1]} columns")

# First few rows
print("\n📋 First 5 Rows:")
print(df_sales.head())

# Last few rows
print("\n📋 Last 5 Rows:")
print(df_sales.tail())

# Column information
print("\n📑 Column Information:")
print(df_sales.info())

# Data types
print("\n🔤 Data Types:")
print(df_sales.dtypes)

# ============================================================================
# 6. BASIC STATISTICS
# ============================================================================

print("\n" + "="*80)
print("BASIC STATISTICAL SUMMARY")
print("="*80)

# Numerical columns summary
print("\n📈 Numerical Columns Statistics:")
print(df_sales.describe())

# Categorical columns summary
print("\n📊 Categorical Columns Summary:")
categorical_cols = df_sales.select_dtypes(include=['object']).columns
for col in categorical_cols[:5]:  # First 5 categorical columns
    print(f"\n{col}:")
    print(df_sales[col].value_counts().head())

# ============================================================================
# 7. DATA QUALITY CHECK
# ============================================================================

print("\n" + "="*80)
print("DATA QUALITY ASSESSMENT")
print("="*80)

# Missing values
print("\n❌ Missing Values:")
missing_data = pd.DataFrame({
    'Column': df_sales.columns,
    'Missing_Count': df_sales.isnull().sum().values,
    'Missing_Percentage': (df_sales.isnull().sum().values / len(df_sales) * 100).round(2)
})
missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
print(missing_data)

# Duplicate rows
duplicates = df_sales.duplicated().sum()
print(f"\n🔄 Duplicate Rows: {duplicates} ({(duplicates/len(df_sales)*100):.2f}%)")

# Unique values per column
print("\n🔢 Unique Values per Column:")
unique_counts = pd.DataFrame({
    'Column': df_sales.columns,
    'Unique_Count': [df_sales[col].nunique() for col in df_sales.columns],
    'Data_Type': df_sales.dtypes.values
})
print(unique_counts)

# ============================================================================
# 8. INITIAL VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("CREATING INITIAL VISUALIZATIONS")
print("="*80)

# Create visualization directory if not exists
os.makedirs('outputs/figures', exist_ok=True)

# 1. Missing values visualization
fig, ax = plt.subplots(figsize=(10, 6))
missing_data_plot = df_sales.isnull().sum().sort_values(ascending=False)[:10]
missing_data_plot.plot(kind='bar', ax=ax, color='coral')
ax.set_title('Top 10 Columns with Missing Values', fontsize=14, fontweight='bold')
ax.set_xlabel('Columns', fontsize=12)
ax.set_ylabel('Number of Missing Values', fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/figures/01_missing_values.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_missing_values.png")
plt.show()

# 2. Data types distribution
fig, ax = plt.subplots(figsize=(8, 6))
dtype_counts = df_sales.dtypes.value_counts()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
ax.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%', 
       startangle=90, colors=colors)
ax.set_title('Distribution of Data Types', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/02_data_types_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_data_types_distribution.png")
plt.show()

# ============================================================================
# 9. SAVE INITIAL INSPECTION REPORT
# ============================================================================

# Create comprehensive report
report = f"""
RETAIL & MARKETING ANALYTICS PROJECT
Initial Data Inspection Report
{'='*80}

1. DATASET OVERVIEW
   - Total Records: {df_sales.shape[0]:,}
   - Total Columns: {df_sales.shape[1]}
   - Memory Usage: {df_sales.memory_usage(deep=True).sum() / 1024**2:.2f} MB
   
2. DATA QUALITY SUMMARY
   - Missing Values: {df_sales.isnull().sum().sum():,} cells ({(df_sales.isnull().sum().sum()/(df_sales.shape[0]*df_sales.shape[1])*100):.2f}%)
   - Duplicate Rows: {duplicates} ({(duplicates/len(df_sales)*100):.2f}%)
   - Complete Rows: {df_sales.dropna().shape[0]:,} ({(df_sales.dropna().shape[0]/len(df_sales)*100):.2f}%)

3. COLUMN TYPES
   - Numerical: {len(df_sales.select_dtypes(include=[np.number]).columns)}
   - Categorical/Object: {len(df_sales.select_dtypes(include=['object']).columns)}
   - DateTime: {len(df_sales.select_dtypes(include=['datetime64']).columns)}

4. NEXT STEPS
   ✓ Proceed to Data Cleaning (Notebook 02)
   - Handle missing values
   - Remove duplicates
   - Convert data types
   - Handle outliers
   - Feature engineering

Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""

# Save report
with open('outputs/reports/01_initial_inspection_report.txt', 'w') as f:
    f.write(report)

print("\n✓ Initial inspection report saved!")
print(report)

# ============================================================================
# 10. SAVE CHECKPOINT
# ============================================================================

# Save initial loaded data for reference
df_sales.to_csv('data/raw/original_data_checkpoint.csv', index=False)
print("\n✓ Data checkpoint saved: data/raw/original_data_checkpoint.csv")

print("\n" + "="*80)
print("NOTEBOOK 01 COMPLETED SUCCESSFULLY!")
print("="*80)
print("\n✅ Next Step: Open Notebook 02 - Data Cleaning and Preprocessing")
print("="*80)
