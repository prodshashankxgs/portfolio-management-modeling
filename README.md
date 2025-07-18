# Portfolio Management PPI Modeling Project

## Project Overview
This project aims to forecast month-on-month percentage changes in the Producer Price Index (PPI) for portfolio management services using machine learning and time series analysis.

**Target Variable**: PPIDF01 Index (PPI Portfolio Management, NSA)  
**Objective**: Predict MoM% changes using market returns, macro risk factors, and fund flows  
**Focus**: Real-time dynamics in fee-based pricing series (no seasonal adjustment)

## Phase One: Data Collection Strategy ✅

### Completed Components

#### 1. **Data Collection Infrastructure**
- `data_collection.py`: Modular data collection class with:
  - FRED API integration with rate limiting
  - Intelligent caching system (1-day cache expiry)
  - Error handling and retry logic
  - Polars-based data processing for performance

#### 2. **Data Sources Implemented**
- **Target Variable**: PPIDF01 (PPI Portfolio Management, NSA)
- **Market Returns**: S&P 500, NASDAQ, Russell 2000, VIX
- **Bond Markets**: 10Y/2Y/3M Treasury yields, corporate spreads
- **Macro Indicators**: Dollar index, oil, gold, unemployment, CPI

#### 3. **Technical Architecture**
- **Environment**: Python virtual environment with dependencies
- **Data Processing**: Polars for high-performance time series operations
- **API Management**: Rate-limited calls with automatic caching
- **Storage**: JSON-based data persistence with DataFrame serialization

#### 4. **Jupyter Notebook**
- `01_data_collection_cleaning.ipynb`: Complete Phase One workflow
- Data quality assessment functions
- Initial visualization of target variable
- Comprehensive data collection pipeline

## Setup Instructions

### 1. Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies (already done)
pip install -r requirements.txt
```

### 2. API Configuration
1. Get a FRED API key from: https://fred.stlouisfed.org/docs/api/api_key.html
2. Create a `.env` file in the project root:
```bash
cp .env.example .env
```
3. Add your API key to `.env`:
```
FRED_API_KEY=your_actual_api_key_here
```

### 3. Run Phase One
```bash
# Start Jupyter notebook
jupyter notebook

# Open and run: 01_data_collection_cleaning.ipynb
```

## Project Structure
```
Portfolio Management Modeling/
├── .env                           # API keys
├── requirements.txt               # Python dependencies
├── data_collection.py             # Core data collection module
├── 01_data_collection_cleaning.ipynb  # Phase One notebook
├── data_cache/                    # Cached API responses
├── project_outline.txt            # Original project outline
├── in-depth_itinerary.txt         # Detailed project phases
└── README.md                      # This file
```

## Key Features

### Data Collection Module (`data_collection.py`)
- **Modular Design**: Separate methods for different data categories
- **Caching**: Automatic 1-day cache to avoid redundant API calls
- **Rate Limiting**: Respects API rate limits (100ms between calls)
- **Error Handling**: Graceful handling of missing data and API errors
- **Polars Integration**: High-performance DataFrame operations

### Data Sources
| Category | Series | Description |
|----------|--------|-------------|
| Target | PPIDF01 | PPI Portfolio Management (NSA) |
| Market | SP500, NASDAQCOM, VIXCLS | Stock indices and volatility |
| Bonds | DGS10, DGS2, DGS3MO | Treasury yield curve |
| Macro | DTWEXBGS, DCOILWTICO, UNRATE | Dollar, oil, employment |

## Next Steps: Phase Two

### Upcoming Components
1. **Exploratory Data Analysis**
   - Correlation analysis between PPI and predictors
   - Lead-lag relationship identification
   - Regime-dependent behavior analysis
   - Structural break testing

2. **Feature Engineering**
   - Lag structure optimization (1-12 months)
   - Rolling volatilities and momentum indicators
   - Regime indicators and interaction terms
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
