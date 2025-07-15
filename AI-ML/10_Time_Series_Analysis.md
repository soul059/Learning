# 10. Time Series Analysis

## 🎯 Learning Objectives
- Understand time series data characteristics and patterns
- Master traditional time series forecasting methods
- Learn modern deep learning approaches for temporal data
- Apply time series analysis to real-world problems

---

## 1. Introduction to Time Series

**Time Series** is a sequence of data points indexed in temporal order, where observations are collected at regular intervals.

### 1.1 What is Time Series Analysis? 🟢

#### Key Characteristics:
- **Temporal ordering**: Order of observations matters
- **Trend**: Long-term increase or decrease in data
- **Seasonality**: Regular patterns that repeat over known periods
- **Cyclical patterns**: Fluctuations with no fixed period
- **Irregularity/Noise**: Random variation in data

#### Types of Time Series:
- **Univariate**: Single variable over time
- **Multivariate**: Multiple variables over time
- **Regular**: Equal time intervals between observations
- **Irregular**: Unequal time intervals

#### Applications:
- **Finance**: Stock prices, exchange rates, portfolio optimization
- **Economics**: GDP, inflation, unemployment rates
- **Weather**: Temperature, rainfall, atmospheric pressure
- **Sales**: Revenue forecasting, demand planning
- **IoT**: Sensor data, system monitoring
- **Healthcare**: Patient monitoring, epidemic modeling

### 1.2 Time Series Components 🟢

#### Decomposition Example:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

def create_sample_time_series(n_points=365*3, noise_level=0.1):
    """Create sample time series with trend, seasonality, and noise"""
    
    # Create date range
    dates = pd.date_range('2021-01-01', periods=n_points, freq='D')
    
    # Create components
    trend = np.linspace(100, 150, n_points)  # Linear trend
    
    # Seasonal component (yearly and weekly)
    seasonal_yearly = 10 * np.sin(2 * np.pi * np.arange(n_points) / 365.25)
    seasonal_weekly = 3 * np.sin(2 * np.pi * np.arange(n_points) / 7)
    seasonal = seasonal_yearly + seasonal_weekly
    
    # Noise
    noise = np.random.normal(0, noise_level * trend.mean(), n_points)
    
    # Combine components
    ts_additive = trend + seasonal + noise
    ts_multiplicative = trend * (1 + seasonal/100) * (1 + noise/100)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'trend': trend,
        'seasonal': seasonal,
        'noise': noise,
        'additive': ts_additive,
        'multiplicative': ts_multiplicative
    })
    
    df.set_index('date', inplace=True)
    return df

# Create and visualize sample data
def visualize_time_series_components(df):
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Original time series
    axes[0, 0].plot(df.index, df['additive'], color='blue', alpha=0.7)
    axes[0, 0].set_title('Additive Time Series')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(df.index, df['multiplicative'], color='red', alpha=0.7)
    axes[0, 1].set_title('Multiplicative Time Series')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Trend
    axes[1, 0].plot(df.index, df['trend'], color='green', linewidth=2)
    axes[1, 0].set_title('Trend Component')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Seasonal
    axes[1, 1].plot(df.index[:365], df['seasonal'][:365], color='orange', linewidth=2)
    axes[1, 1].set_title('Seasonal Component (First Year)')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Noise
    axes[2, 0].plot(df.index, df['noise'], color='purple', alpha=0.5)
    axes[2, 0].set_title('Noise Component')
    axes[2, 0].set_ylabel('Value')
    axes[2, 0].set_xlabel('Date')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Combined view
    axes[2, 1].plot(df.index, df['trend'], label='Trend', linewidth=2)
    axes[2, 1].plot(df.index, df['additive'], label='Complete Series', alpha=0.7)
    axes[2, 1].set_title('Trend vs Complete Series')
    axes[2, 1].set_ylabel('Value')
    axes[2, 1].set_xlabel('Date')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Generate and visualize sample data
df_sample = create_sample_time_series()
visualize_time_series_components(df_sample)
```

#### Formal Decomposition:
```python
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def decompose_time_series(ts, model='additive', period=365):
    """Decompose time series into trend, seasonal, and residual components"""
    
    decomposition = seasonal_decompose(ts, model=model, period=period)
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Original
    decomposition.observed.plot(ax=axes[0], title='Original Time Series')
    axes[0].grid(True, alpha=0.3)
    
    # Trend
    decomposition.trend.plot(ax=axes[1], title='Trend', color='red')
    axes[1].grid(True, alpha=0.3)
    
    # Seasonal
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal', color='green')
    axes[2].grid(True, alpha=0.3)
    
    # Residual
    decomposition.resid.plot(ax=axes[3], title='Residual', color='purple')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return decomposition

# Apply decomposition
decomposition = decompose_time_series(df_sample['additive'])
```

---

## 2. Exploratory Time Series Analysis

### 2.1 Stationarity 🟡

#### Augmented Dickey-Fuller Test:
```python
def check_stationarity(ts, significance_level=0.05):
    """Check if time series is stationary using ADF test"""
    
    # Perform ADF test
    result = adfuller(ts.dropna())
    
    print('Augmented Dickey-Fuller Test Results:')
    print(f'ADF Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.3f}')
    
    # Interpretation
    if result[1] <= significance_level:
        print(f'\nResult: Time series is stationary (reject null hypothesis)')
        print(f'p-value ({result[1]:.6f}) <= significance level ({significance_level})')
    else:
        print(f'\nResult: Time series is non-stationary (fail to reject null hypothesis)')
        print(f'p-value ({result[1]:.6f}) > significance level ({significance_level})')
    
    return result[1] <= significance_level

# Test stationarity
print("Testing original series:")
is_stationary = check_stationarity(df_sample['additive'])

# Make series stationary through differencing
def make_stationary(ts, max_diff=3):
    """Make time series stationary through differencing"""
    
    diff_ts = ts.copy()
    diff_order = 0
    
    for i in range(max_diff):
        if check_stationarity(diff_ts):
            break
        
        diff_ts = diff_ts.diff().dropna()
        diff_order += 1
        print(f"\nAfter {diff_order} differencing:")
    
    return diff_ts, diff_order

# Apply differencing
stationary_ts, diff_order = make_stationary(df_sample['additive'])
```

### 2.2 Autocorrelation Analysis 🟡

#### ACF and PACF Plots:
```python
def plot_acf_pacf(ts, lags=40):
    """Plot ACF and PACF to analyze autocorrelation"""
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    # ACF plot
    plot_acf(ts.dropna(), lags=lags, ax=axes[0], alpha=0.05)
    axes[0].set_title('Autocorrelation Function (ACF)')
    axes[0].grid(True, alpha=0.3)
    
    # PACF plot
    plot_pacf(ts.dropna(), lags=lags, ax=axes[1], alpha=0.05)
    axes[1].set_title('Partial Autocorrelation Function (PACF)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Plot ACF and PACF for original and stationary series
print("ACF/PACF for original series:")
plot_acf_pacf(df_sample['additive'])

print("ACF/PACF for stationary series:")
plot_acf_pacf(stationary_ts)
```

#### Manual Autocorrelation Calculation:
```python
def calculate_autocorrelation(ts, max_lag=20):
    """Calculate autocorrelation manually"""
    
    ts_clean = ts.dropna()
    n = len(ts_clean)
    mean_ts = ts_clean.mean()
    
    autocorrelations = []
    
    for lag in range(max_lag + 1):
        if lag == 0:
            autocorr = 1.0
        else:
            numerator = np.sum((ts_clean[:-lag] - mean_ts) * (ts_clean[lag:] - mean_ts))
            denominator = np.sum((ts_clean - mean_ts) ** 2)
            autocorr = numerator / denominator
        
        autocorrelations.append(autocorr)
    
    return autocorrelations

# Calculate and plot manual autocorrelation
manual_acf = calculate_autocorrelation(df_sample['additive'], max_lag=20)

plt.figure(figsize=(12, 6))
plt.plot(range(len(manual_acf)), manual_acf, 'bo-')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='5% significance')
plt.axhline(y=-0.05, color='r', linestyle='--', alpha=0.5)
plt.title('Manual Autocorrelation Calculation')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
```

---

## 3. Traditional Time Series Models

### 3.1 Moving Averages 🟢

#### Simple Moving Average:
```python
def simple_moving_average(ts, window_size):
    """Calculate simple moving average"""
    return ts.rolling(window=window_size).mean()

def weighted_moving_average(ts, weights):
    """Calculate weighted moving average"""
    window_size = len(weights)
    weights = np.array(weights) / np.sum(weights)  # Normalize weights
    
    wma = []
    for i in range(len(ts)):
        if i < window_size - 1:
            wma.append(np.nan)
        else:
            wma.append(np.sum(ts.iloc[i-window_size+1:i+1] * weights))
    
    return pd.Series(wma, index=ts.index)

def exponential_moving_average(ts, alpha=0.3):
    """Calculate exponential moving average"""
    ema = [ts.iloc[0]]  # Initialize with first value
    
    for i in range(1, len(ts)):
        ema.append(alpha * ts.iloc[i] + (1 - alpha) * ema[i-1])
    
    return pd.Series(ema, index=ts.index)

# Apply different moving averages
ts = df_sample['additive']
sma_5 = simple_moving_average(ts, 5)
sma_20 = simple_moving_average(ts, 20)
wma = weighted_moving_average(ts, [0.1, 0.2, 0.3, 0.4])  # More weight to recent values
ema = exponential_moving_average(ts, alpha=0.3)

# Plot comparisons
plt.figure(figsize=(15, 8))
plt.plot(ts.index, ts, label='Original', alpha=0.7)
plt.plot(sma_5.index, sma_5, label='SMA(5)', linewidth=2)
plt.plot(sma_20.index, sma_20, label='SMA(20)', linewidth=2)
plt.plot(wma.index, wma, label='WMA(4)', linewidth=2)
plt.plot(ema.index, ema, label='EMA(α=0.3)', linewidth=2)

plt.title('Comparison of Moving Averages')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 3.2 ARIMA Models 🟡

#### ARIMA Implementation:
```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ARIMAModel:
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None
        self.fitted_model = None
    
    def fit(self, ts):
        """Fit ARIMA model to time series"""
        self.model = ARIMA(ts, order=self.order)
        self.fitted_model = self.model.fit()
        return self.fitted_model
    
    def forecast(self, steps=1, alpha=0.05):
        """Generate forecasts"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        forecast_result = self.fitted_model.forecast(steps=steps, alpha=alpha)
        
        if hasattr(forecast_result, 'predicted_mean'):
            # statsmodels >= 0.12
            forecasts = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int()
        else:
            # older statsmodels versions
            forecasts = forecast_result
            conf_int = None
        
        return forecasts, conf_int
    
    def diagnostic_plots(self):
        """Generate diagnostic plots"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before generating diagnostics")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Residuals plot
        residuals = self.fitted_model.resid
        axes[0, 0].plot(residuals)
        axes[0, 0].set_title('Residuals')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # ACF of residuals
        plot_acf(residuals, ax=axes[1, 0], lags=20)
        axes[1, 0].set_title('ACF of Residuals')
        
        # Histogram of residuals
        axes[1, 1].hist(residuals, bins=20, density=True, alpha=0.7)
        axes[1, 1].set_title('Histogram of Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def model_summary(self):
        """Print model summary"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before generating summary")
        
        print(self.fitted_model.summary())
        
        # Ljung-Box test for residual autocorrelation
        lb_test = acorr_ljungbox(self.fitted_model.resid, lags=10, return_df=True)
        print("\nLjung-Box Test (p-values should be > 0.05 for good model):")
        print(lb_test)

# Auto ARIMA order selection
def find_best_arima_order(ts, max_p=3, max_d=2, max_q=3):
    """Find best ARIMA order using AIC criterion"""
    
    best_aic = float('inf')
    best_order = None
    
    results = []
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(ts, order=(p, d, q))
                    fitted_model = model.fit()
                    aic = fitted_model.aic
                    
                    results.append({
                        'order': (p, d, q),
                        'aic': aic,
                        'bic': fitted_model.bic,
                        'hqic': fitted_model.hqic
                    })
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                        
                except Exception as e:
                    continue
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('aic')
    
    print(f"Best ARIMA order: {best_order} (AIC: {best_aic:.2f})")
    print("\nTop 5 models by AIC:")
    print(results_df.head())
    
    return best_order, results_df

# Apply ARIMA modeling
ts_train = df_sample['additive'][:-30]  # Keep last 30 for testing
ts_test = df_sample['additive'][-30:]

# Find best order
best_order, model_comparison = find_best_arima_order(ts_train)

# Fit best model
arima_model = ARIMAModel(order=best_order)
fitted_arima = arima_model.fit(ts_train)

# Generate forecasts
forecasts, conf_int = arima_model.forecast(steps=30, alpha=0.05)

# Plot results
plt.figure(figsize=(15, 8))
plt.plot(ts_train.index, ts_train, label='Training Data', color='blue')
plt.plot(ts_test.index, ts_test, label='Actual Test Data', color='green')
plt.plot(ts_test.index, forecasts, label='ARIMA Forecast', color='red', linestyle='--')

if conf_int is not None:
    plt.fill_between(ts_test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], 
                     color='red', alpha=0.2, label='Confidence Interval')

plt.title(f'ARIMA{best_order} Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Calculate metrics
mse = mean_squared_error(ts_test, forecasts)
mae = mean_absolute_error(ts_test, forecasts)
rmse = np.sqrt(mse)

print(f"\nForecast Accuracy Metrics:")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# Model diagnostics
arima_model.diagnostic_plots()
arima_model.model_summary()
```

### 3.3 Seasonal ARIMA (SARIMA) 🟡

#### SARIMA Implementation:
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

class SARIMAModel:
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
    
    def fit(self, ts):
        """Fit SARIMA model"""
        self.model = SARIMAX(ts, order=self.order, seasonal_order=self.seasonal_order)
        self.fitted_model = self.model.fit(disp=False)
        return self.fitted_model
    
    def forecast(self, steps=1, alpha=0.05):
        """Generate forecasts"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        forecast_result = self.fitted_model.get_forecast(steps=steps, alpha=alpha)
        forecasts = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()
        
        return forecasts, conf_int

def find_best_sarima_order(ts, max_p=2, max_d=1, max_q=2, 
                          max_P=1, max_D=1, max_Q=1, s=12):
    """Find best SARIMA order using AIC"""
    
    best_aic = float('inf')
    best_order = None
    best_seasonal_order = None
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                for P in range(max_P + 1):
                    for D in range(max_D + 1):
                        for Q in range(max_Q + 1):
                            try:
                                model = SARIMAX(ts, 
                                              order=(p, d, q),
                                              seasonal_order=(P, D, Q, s))
                                fitted_model = model.fit(disp=False)
                                aic = fitted_model.aic
                                
                                if aic < best_aic:
                                    best_aic = aic
                                    best_order = (p, d, q)
                                    best_seasonal_order = (P, D, Q, s)
                                    
                            except Exception as e:
                                continue
    
    print(f"Best SARIMA order: {best_order} x {best_seasonal_order} (AIC: {best_aic:.2f})")
    return best_order, best_seasonal_order

# Apply SARIMA modeling
best_order, best_seasonal_order = find_best_sarima_order(ts_train, s=365)

sarima_model = SARIMAModel(order=best_order, seasonal_order=best_seasonal_order)
fitted_sarima = sarima_model.fit(ts_train)

# Generate forecasts
sarima_forecasts, sarima_conf_int = sarima_model.forecast(steps=30, alpha=0.05)

# Plot SARIMA results
plt.figure(figsize=(15, 8))
plt.plot(ts_train.index, ts_train, label='Training Data', color='blue')
plt.plot(ts_test.index, ts_test, label='Actual Test Data', color='green')
plt.plot(ts_test.index, sarima_forecasts, label='SARIMA Forecast', color='red', linestyle='--')
plt.fill_between(ts_test.index, sarima_conf_int.iloc[:, 0], sarima_conf_int.iloc[:, 1], 
                 color='red', alpha=0.2, label='Confidence Interval')

plt.title(f'SARIMA{best_order}x{best_seasonal_order} Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Compare ARIMA vs SARIMA
sarima_mse = mean_squared_error(ts_test, sarima_forecasts)
sarima_mae = mean_absolute_error(ts_test, sarima_forecasts)
sarima_rmse = np.sqrt(sarima_mse)

print(f"\nSARIMA Forecast Accuracy Metrics:")
print(f"MSE: {sarima_mse:.4f}")
print(f"MAE: {sarima_mae:.4f}")
print(f"RMSE: {sarima_rmse:.4f}")

print(f"\nComparison:")
print(f"ARIMA RMSE: {rmse:.4f}")
print(f"SARIMA RMSE: {sarima_rmse:.4f}")
print(f"Improvement: {((rmse - sarima_rmse)/rmse)*100:.2f}%")
```

---

## 4. Modern Deep Learning Approaches

### 4.1 Recurrent Neural Networks (RNNs) 🔴

#### LSTM for Time Series:
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length=20):
        self.data = data
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length]
        return torch.FloatTensor(x), torch.FloatTensor([y])

class LSTMTimeSeriesModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMTimeSeriesModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take only the last output
        out = self.fc(out[:, -1, :])
        
        return out

class GRUTimeSeriesModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super(GRUTimeSeriesModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

def prepare_time_series_data(ts, sequence_length=20, train_ratio=0.8):
    """Prepare time series data for deep learning"""
    
    # Normalize data
    scaler = MinMaxScaler()
    ts_scaled = scaler.fit_transform(ts.values.reshape(-1, 1)).flatten()
    
    # Split train/test
    train_size = int(len(ts_scaled) * train_ratio)
    train_data = ts_scaled[:train_size]
    test_data = ts_scaled[train_size:]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, sequence_length)
    test_dataset = TimeSeriesDataset(test_data, sequence_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader, scaler

def train_deep_model(model, train_loader, val_loader, epochs=100, learning_rate=0.001):
    """Train deep learning time series model"""
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x.unsqueeze(-1))
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x.unsqueeze(-1))
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        if epoch % 20 == 0:
            print(f'Epoch [{epoch}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    
    return train_losses, val_losses

# Prepare data
ts = df_sample['additive']
train_loader, test_loader, scaler = prepare_time_series_data(ts, sequence_length=30)

# Train LSTM model
lstm_model = LSTMTimeSeriesModel(input_size=1, hidden_size=50, num_layers=2)
lstm_train_losses, lstm_val_losses = train_deep_model(lstm_model, train_loader, test_loader, epochs=100)

# Train GRU model
gru_model = GRUTimeSeriesModel(input_size=1, hidden_size=50, num_layers=2)
gru_train_losses, gru_val_losses = train_deep_model(gru_model, train_loader, test_loader, epochs=100)

# Plot training curves
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(lstm_train_losses, label='LSTM Train')
plt.plot(lstm_val_losses, label='LSTM Validation')
plt.title('LSTM Training Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(gru_train_losses, label='GRU Train')
plt.plot(gru_val_losses, label='GRU Validation')
plt.title('GRU Training Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 4.2 Transformer Models for Time Series 🔴

#### Time Series Transformer:
```python
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=1000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=8, num_layers=3, 
                 sequence_length=50, output_size=1, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        
        self.d_model = d_model
        self.sequence_length = sequence_length
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, sequence_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size)
        )
    
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Use last token for prediction
        x = x[:, -1, :]
        
        # Output projection
        output = self.output_projection(x)
        
        return output

# Train Transformer model
transformer_model = TimeSeriesTransformer(
    input_size=1, 
    d_model=64, 
    nhead=8, 
    num_layers=3, 
    sequence_length=30
)

transformer_train_losses, transformer_val_losses = train_deep_model(
    transformer_model, train_loader, test_loader, epochs=100
)

# Plot comparison
plt.figure(figsize=(12, 6))
plt.plot(lstm_val_losses, label='LSTM Validation Loss')
plt.plot(gru_val_losses, label='GRU Validation Loss')
plt.plot(transformer_val_losses, label='Transformer Validation Loss')
plt.title('Model Comparison - Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 5. Advanced Techniques

### 5.1 Facebook Prophet 🟡

#### Prophet Implementation:
```python
# Note: Uncomment to install Prophet
# !pip install prophet

try:
    from prophet import Prophet
    prophet_available = True
except ImportError:
    prophet_available = False
    print("Prophet not available. Install with: pip install prophet")

if prophet_available:
    def apply_prophet(ts, periods=30):
        """Apply Facebook Prophet for time series forecasting"""
        
        # Prepare data for Prophet
        df_prophet = pd.DataFrame({
            'ds': ts.index,
            'y': ts.values
        })
        
        # Create and fit model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode='additive'
        )
        
        model.fit(df_prophet)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        # Plot results
        fig1 = model.plot(forecast)
        plt.title('Prophet Forecast')
        plt.show()
        
        # Plot components
        fig2 = model.plot_components(forecast)
        plt.show()
        
        return model, forecast
    
    # Apply Prophet
    prophet_model, prophet_forecast = apply_prophet(ts_train, periods=30)
    
    # Extract forecasts for test period
    prophet_test_forecasts = prophet_forecast.tail(30)['yhat'].values
    
    # Calculate Prophet metrics
    prophet_mse = mean_squared_error(ts_test, prophet_test_forecasts)
    prophet_rmse = np.sqrt(prophet_mse)
    
    print(f"Prophet RMSE: {prophet_rmse:.4f}")
```

### 5.2 Ensemble Methods 🟡

#### Time Series Ensemble:
```python
class TimeSeriesEnsemble:
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.scaler = MinMaxScaler()
        
    def add_model(self, name, model, weight=1.0):
        """Add a model to the ensemble"""
        self.models[name] = model
        self.weights[name] = weight
        
    def fit(self, ts):
        """Fit all models in the ensemble"""
        self.scaler.fit(ts.values.reshape(-1, 1))
        
        for name, model in self.models.items():
            if hasattr(model, 'fit'):
                model.fit(ts)
    
    def predict(self, steps=1):
        """Generate ensemble predictions"""
        predictions = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'forecast'):
                pred, _ = model.forecast(steps=steps)
                predictions[name] = pred
            elif hasattr(model, 'predict'):
                pred = model.predict(steps)
                predictions[name] = pred
        
        # Weighted average
        ensemble_pred = np.zeros(steps)
        total_weight = sum(self.weights.values())
        
        for name, pred in predictions.items():
            weight = self.weights[name] / total_weight
            ensemble_pred += weight * pred
        
        return ensemble_pred, predictions

# Create ensemble
ensemble = TimeSeriesEnsemble()

# Add models with different weights
ensemble.add_model('ARIMA', arima_model, weight=0.3)
ensemble.add_model('SARIMA', sarima_model, weight=0.4)

# Simple exponential smoothing as baseline
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ses_model = ExponentialSmoothing(ts_train, trend=None, seasonal=None)
fitted_ses = ses_model.fit()

class SESWrapper:
    def __init__(self, fitted_model):
        self.fitted_model = fitted_model
    
    def forecast(self, steps=1):
        return self.fitted_model.forecast(steps), None

ses_wrapper = SESWrapper(fitted_ses)
ensemble.add_model('SES', ses_wrapper, weight=0.3)

# Fit ensemble
ensemble.fit(ts_train)

# Generate ensemble predictions
ensemble_pred, individual_preds = ensemble.predict(steps=30)

# Calculate ensemble metrics
ensemble_mse = mean_squared_error(ts_test, ensemble_pred)
ensemble_rmse = np.sqrt(ensemble_mse)

print(f"Ensemble RMSE: {ensemble_rmse:.4f}")

# Plot ensemble results
plt.figure(figsize=(15, 8))
plt.plot(ts_train.index, ts_train, label='Training Data', color='blue')
plt.plot(ts_test.index, ts_test, label='Actual Test Data', color='green')
plt.plot(ts_test.index, ensemble_pred, label='Ensemble Forecast', color='purple', linewidth=2)

# Plot individual model predictions
colors = ['red', 'orange', 'brown']
for i, (name, pred) in enumerate(individual_preds.items()):
    plt.plot(ts_test.index, pred, label=f'{name} Forecast', 
             color=colors[i % len(colors)], linestyle='--', alpha=0.7)

plt.title('Ensemble vs Individual Model Forecasts')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Model performance comparison
print("\nModel Performance Comparison (RMSE):")
print(f"ARIMA: {rmse:.4f}")
print(f"SARIMA: {sarima_rmse:.4f}")
if prophet_available:
    print(f"Prophet: {prophet_rmse:.4f}")
print(f"Ensemble: {ensemble_rmse:.4f}")
```

---

## 6. Evaluation and Validation

### 6.1 Time Series Cross-Validation 🟡

#### Walk-Forward Validation:
```python
def time_series_cv(ts, n_splits=5, test_size=30):
    """Perform time series cross-validation"""
    
    total_size = len(ts)
    train_size = total_size - test_size
    step_size = train_size // n_splits
    
    splits = []
    
    for i in range(n_splits):
        train_start = i * step_size
        train_end = train_start + train_size - (n_splits - i - 1) * step_size
        test_start = train_end
        test_end = test_start + test_size
        
        if test_end <= total_size:
            train_indices = range(train_start, train_end)
            test_indices = range(test_start, test_end)
            splits.append((train_indices, test_indices))
    
    return splits

def evaluate_model_cv(ts, model_class, model_params, cv_splits):
    """Evaluate model using cross-validation"""
    
    scores = []
    
    for train_idx, test_idx in cv_splits:
        train_data = ts.iloc[train_idx]
        test_data = ts.iloc[test_idx]
        
        # Fit model
        model = model_class(**model_params)
        model.fit(train_data)
        
        # Forecast
        forecasts, _ = model.forecast(steps=len(test_data))
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(test_data, forecasts))
        scores.append(rmse)
    
    return scores

# Perform cross-validation
cv_splits = time_series_cv(ts_train, n_splits=5, test_size=20)

# Evaluate ARIMA
arima_cv_scores = evaluate_model_cv(ts_train, ARIMAModel, 
                                   {'order': best_order}, cv_splits)

# Evaluate SARIMA
sarima_cv_scores = evaluate_model_cv(ts_train, SARIMAModel, 
                                   {'order': best_order, 'seasonal_order': best_seasonal_order}, 
                                   cv_splits)

print("Cross-Validation Results (RMSE):")
print(f"ARIMA - Mean: {np.mean(arima_cv_scores):.4f}, Std: {np.std(arima_cv_scores):.4f}")
print(f"SARIMA - Mean: {np.mean(sarima_cv_scores):.4f}, Std: {np.std(sarima_cv_scores):.4f}")

# Plot CV scores
plt.figure(figsize=(10, 6))
plt.boxplot([arima_cv_scores, sarima_cv_scores], labels=['ARIMA', 'SARIMA'])
plt.title('Cross-Validation RMSE Scores')
plt.ylabel('RMSE')
plt.grid(True, alpha=0.3)
plt.show()
```

### 6.2 Evaluation Metrics 🟢

#### Comprehensive Metrics:
```python
def calculate_forecast_metrics(actual, predicted):
    """Calculate comprehensive forecast evaluation metrics"""
    
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Basic metrics
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    
    # Percentage errors
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    smape = np.mean(2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted))) * 100
    
    # Relative metrics
    naive_forecast = actual[:-1]  # Use previous value as forecast
    naive_mse = mean_squared_error(actual[1:], naive_forecast)
    mase = mae / np.mean(np.abs(np.diff(actual)))  # Mean Absolute Scaled Error
    
    # Direction accuracy
    actual_direction = np.diff(actual) > 0
    predicted_direction = np.diff(predicted) > 0
    direction_accuracy = np.mean(actual_direction == predicted_direction) * 100
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'SMAPE': smape,
        'MASE': mase,
        'Direction Accuracy': direction_accuracy
    }
    
    return metrics

# Calculate metrics for all models
models_forecasts = {
    'ARIMA': forecasts,
    'SARIMA': sarima_forecasts,
    'Ensemble': ensemble_pred
}

print("Comprehensive Evaluation Metrics:")
print("-" * 60)

for model_name, model_forecasts in models_forecasts.items():
    metrics = calculate_forecast_metrics(ts_test.values, model_forecasts)
    print(f"\n{model_name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

# Residual analysis
def analyze_residuals(actual, predicted, model_name):
    """Analyze forecast residuals"""
    
    residuals = actual - predicted
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Residuals plot
    axes[0, 0].plot(residuals)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_title(f'{model_name} - Residuals')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[0, 1].hist(residuals, bins=15, density=True, alpha=0.7)
    axes[0, 1].set_title(f'{model_name} - Residuals Distribution')
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title(f'{model_name} - Q-Q Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Actual vs Predicted
    axes[1, 1].scatter(actual, predicted, alpha=0.6)
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--')
    axes[1, 1].set_xlabel('Actual')
    axes[1, 1].set_ylabel('Predicted')
    axes[1, 1].set_title(f'{model_name} - Actual vs Predicted')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Statistical tests
    from scipy.stats import shapiro, jarque_bera
    
    print(f"\n{model_name} Residual Analysis:")
    print(f"Mean: {residuals.mean():.6f}")
    print(f"Std: {residuals.std():.6f}")
    
    # Normality tests
    shapiro_stat, shapiro_p = shapiro(residuals)
    jb_stat, jb_p = jarque_bera(residuals)
    
    print(f"Shapiro-Wilk test: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
    print(f"Jarque-Bera test: statistic={jb_stat:.4f}, p-value={jb_p:.4f}")

# Analyze residuals for best model
analyze_residuals(ts_test.values, ensemble_pred, 'Ensemble')
```

---

## 🎯 Key Takeaways

### Time Series Analysis Workflow:
1. **Data Exploration**: Understand patterns, trends, seasonality
2. **Stationarity Testing**: Check and achieve stationarity if needed
3. **Model Selection**: Choose appropriate model based on data characteristics
4. **Model Fitting**: Train model with proper parameter tuning
5. **Validation**: Use time series specific validation techniques
6. **Forecasting**: Generate predictions with uncertainty quantification
7. **Evaluation**: Assess model performance using multiple metrics

### Model Selection Guide:
- **Simple patterns**: Moving averages, Exponential smoothing
- **Linear trends**: ARIMA models
- **Seasonal patterns**: SARIMA, seasonal decomposition
- **Complex non-linear**: LSTM, GRU, Transformers
- **Multiple seasonalities**: Prophet, deep learning
- **Uncertainty**: Ensemble methods

### Best Practices:
- Always plot your data first
- Check for stationarity before modeling
- Use proper validation techniques (no data leakage)
- Consider multiple models and ensemble them
- Evaluate on multiple metrics, not just accuracy
- Account for prediction uncertainty

---

## 📚 Next Steps

Continue your journey with:
- **[Ensemble Methods](11_Ensemble_Methods.md)** - Learn to combine multiple models
- **[Model Evaluation & Selection](12_Model_Evaluation_Selection.md)** - Advanced evaluation techniques

---

*Next: [Ensemble Methods →](11_Ensemble_Methods.md)*
