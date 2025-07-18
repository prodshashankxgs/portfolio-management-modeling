# Portfolio Management PPI Modeling Project

## Project Overview
This project aims to forecast month-on-month percentage changes in the Producer Price Index (PPI) for portfolio management services using machine learning and time series analysis.

**Target Variable**: PPIDF01 Index (PPI Portfolio Management, NSA)  
**Objective**: Predict MoM% changes using market returns, macro risk factors, and fund flows  
**Focus**: Real-time dynamics in fee-based pricing series (no seasonal adjustment)

## Data Sources
- **Primary Target**: PCU5239252392 (PPI Portfolio Management Services) from FRED
- **Predictors**: Market returns, bond yields, fund flows, macro indicators
- **APIs**: FRED API, BLS API for economic data

## Project Structure
```
├── data_collection.py              # Modular data collection with caching
├── exploratory_analysis.py         # EDA and feature engineering (Polars)
├── modeling_framework.py           # ML models with time series validation
├── 01_data_collection_polars.ipynb # Data collection notebook
├── 03_modeling_validation.ipynb    # Modeling and validation notebook
├── .env                            # API keys (create from .env.example)
├── requirements.txt                # Essential dependencies only
├── data_cache/                     # Cached data storage
└── README.md                       # This file
```

## Setup Instructions

### 1. Environment Setup
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. API Configuration
Create a `.env` file with your API keys:
```
FRED_API_KEY=your_fred_api_key_here
BLS_API_KEY=your_bls_api_key_here
```

Get your FRED API key from: https://fred.stlouisfed.org/docs/api/api_key.html

### 3. Start Jupyter Notebook
```bash
jupyter notebook
```

## Key Features

### Pure Polars Implementation
- **2-5x faster** than pandas for large datasets
- **Memory efficient** with lazy evaluation
- **Type safe** with compile-time optimizations
- **Modern syntax** with method chaining

### Comprehensive Data Pipeline
- **Automated caching** to minimize API calls
- **Rate limiting** to respect API constraints
- **Error handling** with retry mechanisms
- **Data validation** and quality checks

### Advanced Modeling Framework
- **Time series cross-validation** with expanding windows
- **Multiple model types**: AR, Ridge, Random Forest
- **Feature engineering** with lagged variables
- **Performance metrics** including directional accuracy

## Getting Started

### 1. Data Collection
Open `01_data_collection_polars.ipynb` to:
- Fetch PPI data using the correct ticker (PCU5239252392)
- Collect predictor variables from FRED/BLS APIs
- Cache data for efficient reuse
- Perform initial data quality checks
   - Temporal features (month/quarter effects)

3. **Initial Modeling**
   - Baseline autoregressive models
   - Linear models with regularization
   - Time series cross-validation framework

## Technical Notes

### Performance Optimizations
- **Polars**: 2-5x faster than pandas for time series operations
- **Caching**: Reduces API calls and improves development speed
- **Modular Design**: Easy to extend with additional data sources

### Data Quality Considerations
- **Missing Values**: Handled gracefully with null checks
- **Frequency Alignment**: All data converted to monthly frequency
- **Revision Patterns**: Economic data revisions tracked
- **Release Schedules**: Different data release timing considered

## Troubleshooting

### Common Issues
1. **API Key Error**: Ensure FRED_API_KEY is set in `.env` file
2. **Rate Limiting**: Built-in 100ms delays between API calls
3. **Cache Issues**: Delete `data_cache/` folder to force fresh data
4. **Missing Data**: Some series may have limited historical availability

### Data Availability
- **PPIDF01**: Available from ~2007 (limited historical data)
- **Market Data**: Extensive historical coverage (2000+)
- **Macro Data**: Varies by series, most available from 1990s+

## Contact & Support
For questions about the data collection infrastructure or API issues, refer to:
- FRED API Documentation: https://fred.stlouisfed.org/docs/api/
- BLS API Documentation: https://www.bls.gov/developers/

---

**Phase One Status**: ✅ COMPLETE  
**Next Phase**: Exploratory Data Analysis  
**Last Updated**: July 18, 2025
