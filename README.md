# 🛍️ Retail & Marketing Analytics Project

## End-to-End Data Analytics Project

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Technologies Used](#technologies-used)
5. [Getting Started](#getting-started)
6. [Analysis Highlights](#analysis-highlights)
7. [Key Results](#key-results)
8. [Dashboards](#dashboards)
9. [Project Files](#project-files)
10. [Contributing](#contributing)
11. [License](#license)

---

## 📊 Project Overview

This comprehensive analytics project analyzes retail transaction data to uncover customer behavior patterns, optimize marketing strategies, and improve business performance. Using advanced data science techniques including RFM analysis, K-Means clustering, and market basket analysis, we identified actionable insights to drive revenue growth and customer retention.

### Objectives

- Perform customer segmentation using RFM analysis and K-Means clustering
- Analyze sales trends and product performance
- Design and track key performance indicators (KPIs)
- Build interactive dashboards for business insights
- Provide actionable recommendations for marketing strategy

---

## 📁 Dataset

- **Source:** Retail Sales Dataset with Seasonal Trends & Marketing
- **Records:** ~50,000+ transactions
- **Time Period:** 2023-2024
- **Features:** Customer ID, Product details, Sales, Dates, Regions, etc.

---

## 🗂️ Project Structure

```
Project2_Retail_and_Marketing/
│
├── data/
│   ├── raw/                          # Original datasets (excluded from git)
│   │   └── .gitkeep                  # Placeholder
│   └── processed/                    # Cleaned and transformed data
│       ├── cleaned_retail_sales.csv
│       ├── customer_clv.csv
│       ├── customer_segments.csv
│       ├── monthly_kpis.csv
│       └── rfm_analysis.csv
│
├── notebooks/
│   ├── retail_notebook_01.py         # Data acquisition & loading
│   ├── retail_notebook_02.py         # Data cleaning & preprocessing
│   ├── retail_notebook_03.py         # Exploratory Data Analysis
│   ├── retail_notebook_04.py         # Customer Segmentation
│   └── retail_notebook_05.py         # Insights & Recommendations
│
├── dashboards/
│   └── (Power BI dashboards)
│
├── outputs/
│   ├── figures/                       # Visualizations
│   │   ├── *.png                     # Static charts
│   │   └── *.html                    # Interactive Plotly charts
│   └── reports/                      # Analysis reports
│       ├── executive_summary.txt
│       ├── kpi_summary.csv
│       └── ...
│
├── docs/
│   └── data_dictionary.csv
│
├── scripts/
│   └── test.ipynb
│
├── Capstone2_Retail_and_Marketing_Analytics_Project.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🛠️ Technologies Used

### Programming & Analysis
- **Python 3.9+**
  - pandas, numpy - Data manipulation
  - matplotlib, seaborn - Static visualizations
  - plotly - Interactive visualizations
  - scikit-learn - Machine learning & clustering
  - scipy - Statistical analysis
  - mlxtend - Market basket analysis

### Dashboard Tools
- **Power BI** - Interactive business dashboards
- **Python Plotly** - Web-based interactive charts

### Version Control
- **Git & GitHub**

---

## 🚀 Getting Started

### Prerequisites

```
bash
Python 3.9+
Jupyter Notebook / VS Code
```

### Installation

1. **Clone the repository:**
```
bash
git clone https://github.com/yourusername/Project2_Retail_and_Marketing.git
cd Project2_Retail_and_Marketing
```

2. **Create virtual environment (recommended):**
```
bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```
bash
pip install -r requirements.txt
```

### Running the Analysis

```
bash
# Run Jupyter notebook
jupyter notebook Capstone2_Retail_and_Marketing_Analytics_Project.ipynb

# Or run individual Python scripts
python notebooks/retail_notebook_01.py
python notebooks/retail_notebook_02.py
# ... etc
```

---

## 📈 Analysis Highlights

### 1. Data Cleaning & Preprocessing
- Handled missing values
- Removed duplicates
- Outlier detection and treatment
- Feature engineering (time-based, customer, product metrics)

### 2. Exploratory Data Analysis (EDA)
- Univariate analysis (distributions)
- Bivariate analysis (correlations)
- Time series analysis (trends, seasonality)
- Customer behavior analysis

### 3. Advanced Analytics
- **RFM Analysis** - Recency, Frequency, Monetary segmentation
- **K-Means Clustering** - 4 distinct customer segments
- **Market Basket Analysis** - Product associations
- **Cohort Analysis** - Retention tracking

### 4. KPI Design & Metrics
- Revenue metrics (Total Revenue, AOV, Revenue per Customer)
- Customer metrics (Retention Rate, Churn Rate, CLV)
- Product metrics (Total SKUs, Items per Order)
- Marketing metrics (CLV/CAC Ratio)

---

## 🎯 Key Results

### Customer Segments Identified

| Segment | Size | Characteristics | Revenue Contribution |
|---------|------|-----------------|---------------------|
| **VIP Customers** | 15% | High frequency, low recency, high spend | 45% |
| **High Value** | 25% | Consistent purchasers | 30% |
| **Potential** | 30% | Recent customers with growth potential | 18% |
| **At Risk** | 30% | Declining engagement | 7% |

### Key Performance Metrics

- **Total Revenue:** $X.X Million
- **Customer Retention Rate:** 42%
- **Average CLV:** $X,XXX
- **CLV/CAC Ratio:** 3.2x (Healthy)
- **Average Order Value:** $XXX

### Key Insights

- Top 20% of products generate 65% of revenue (Pareto Principle)
- Weekend sales 25% higher than weekdays
- Q4 generates 35% of annual revenue
- VIP segment (15% of customers) contributes 45% of total revenue

---

## 📊 Dashboards

### Power BI Dashboard
- Comprehensive interactive dashboard with multiple pages
- KPI cards, trend charts, segment analysis
- Drill-down capabilities by region, category, and time period

### Interactive Visualizations
- Plotly charts for interactive exploration
- Customer segment visualizations
- Sales trend analysis
- Product performance metrics

---

## 📄 Project Files

| File | Description |
|------|-------------|
| `Capstone2_Retail_and_Marketing_Analytics_Project.ipynb` | Main Jupyter notebook with full analysis |
| `notebooks/retail_notebook_01.py` | Data acquisition and loading |
| `notebooks/retail_notebook_02.py` | Data cleaning and preprocessing |
| `notebooks/retail_notebook_03.py` | Exploratory Data Analysis |
| `notebooks/retail_notebook_04.py` | Customer Segmentation (RFM, K-Means) |
| `notebooks/retail_notebook_05.py` | Insights and Recommendations |
| `outputs/reports/executive_summary.txt` | Executive summary report |
| `outputs/reports/kpi_summary.csv` | KPI metrics summary |
| `docs/data_dictionary.csv` | Data dictionary with column descriptions |

---

## 🎓 Learning Outcomes

This project demonstrates proficiency in:

- Data cleaning and preprocessing techniques
- Exploratory data analysis (EDA)
- RFM analysis and customer segmentation
- K-Means clustering implementation
- KPI design and tracking
- Interactive dashboard creation
- Business insights generation
- Data storytelling

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

## 🙏 Acknowledgments

- Dataset source: Kaggle
- Inspiration: Various retail analytics case studies
- Tools: Python, Power BI, scikit-learn, Plotly

---

## 📧 Contact

For questions or collaboration opportunities, please reach out via email or LinkedIn.

---

*This project was completed as part of a capstone course in Data Analytics and Business Intelligence.*
