"""
Energy Consumption Prediction - XGBoost Training Script
Trains model on household power consumption data in Azure ML
"""

import subprocess
import sys

# Install required packages
print("Installing required packages...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                      "xgboost==2.0.2", "mlflow==2.8.0"])

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

import mlflow
import mlflow.xgboost


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--model_output', type=str, default='./outputs', help='Output path for model')
    parser.add_argument('--n_estimators', type=int, default=200, help='Number of boosting rounds')
    parser.add_argument('--max_depth', type=int, default=6, help='Maximum tree depth')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    return parser.parse_args()


def load_and_preprocess_data(data_path):
    """Load and preprocess the household power consumption data"""
    print(f"Loading data from {data_path}...")
    
    # Read the data
    df = pd.read_csv(
        data_path,
        sep=';',
        parse_dates={'datetime': ['Date', 'Time']},
        na_values=['?'],
        low_memory=False
    )
    
    print(f"Raw data shape: {df.shape}")
    
    # Handle missing values
    df = df.dropna()
    
    # Convert to numeric
    numeric_columns = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                      'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    
    print(f"After preprocessing shape: {df.shape}")
    
    return df


def engineer_features(df):
    """Create time-based and lag features"""
    print("Engineering features...")
    
    # Set datetime as index
    df = df.set_index('datetime')
    df = df.sort_index()
    
    # Extract time features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding for hour and month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Lag features
    df['power_lag_1h'] = df['Global_active_power'].shift(60)  # 1 hour ago (1-min intervals)
    df['power_lag_24h'] = df['Global_active_power'].shift(1440)  # 24 hours ago
    df['power_lag_7d'] = df['Global_active_power'].shift(10080)  # 7 days ago
    
    # Rolling statistics
    df['power_roll_mean_24h'] = df['Global_active_power'].rolling(window=1440).mean()
    df['power_roll_std_24h'] = df['Global_active_power'].rolling(window=1440).std()
    df['power_roll_min_24h'] = df['Global_active_power'].rolling(window=1440).min()
    df['power_roll_max_24h'] = df['Global_active_power'].rolling(window=1440).max()
    
    # Drop rows with NaN created by lag/rolling features
    df = df.dropna()
    
    print(f"After feature engineering shape: {df.shape}")
    
    return df


def prepare_train_test(df, test_size=0.2):
    """Prepare training and test sets"""
    print("Preparing train/test split...")
    
    # Define features and target
    feature_cols = [
        'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
        'hour', 'day_of_week', 'day_of_month', 'month', 'quarter', 'is_weekend',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        'power_lag_1h', 'power_lag_24h', 'power_lag_7d',
        'power_roll_mean_24h', 'power_roll_std_24h', 'power_roll_min_24h', 'power_roll_max_24h'
    ]
    
    target_col = 'Global_active_power'
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Train/test split (time-based)
    split_index = int(len(df) * (1 - test_size))
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, feature_cols


def train_model(X_train, y_train, X_test, y_test, n_estimators=200, max_depth=6, learning_rate=0.1):
    """Train XGBoost model"""
    print("Training XGBoost model...")
    
    # Define model
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    
    # Train
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    
    print("Training complete!")
    
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model performance"""
    print("Evaluating model...")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'test_r2': r2_score(y_test, y_test_pred)
    }
    
    print("\n=== Model Performance ===")
    print(f"Train RMSE: {metrics['train_rmse']:.4f}")
    print(f"Train MAE: {metrics['train_mae']:.4f}")
    print(f"Train R²: {metrics['train_r2']:.4f}")
    print(f"Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"Test MAE: {metrics['test_mae']:.4f}")
    print(f"Test R²: {metrics['test_r2']:.4f}")
    print("========================\n")
    
    return metrics


def save_model(model, feature_cols, metrics, output_path):
    """Save model and metadata"""
    print(f"Saving model to {output_path}...")
    
    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / 'model.pkl'
    joblib.dump(model, model_path)
    
    # Save feature columns
    features_path = output_dir / 'features.json'
    with open(features_path, 'w') as f:
        json.dump({'features': feature_cols}, f, indent=2)
    
    # Save metrics
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save metadata
    metadata = {
        'model_type': 'XGBoost',
        'trained_at': datetime.now().isoformat(),
        'n_features': len(feature_cols),
        'metrics': metrics
    }
    
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved successfully to {model_path}")
    
    return model_path


def main():
    """Main training pipeline"""
    args = parse_args()
    
    # Start MLflow run
    mlflow.start_run()
    
    # Log parameters
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("learning_rate", args.learning_rate)
    mlflow.log_param("test_size", args.test_size)
    
    # Load and preprocess data
    df = load_and_preprocess_data(args.data_path)
    
    # Feature engineering
    df = engineer_features(df)
    
    # Prepare train/test split
    X_train, X_test, y_train, y_test, feature_cols = prepare_train_test(df, args.test_size)
    
    # Train model
    model = train_model(X_train, y_train, X_test, y_test, 
                       args.n_estimators, args.max_depth, args.learning_rate)
    
    # Evaluate model
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Log metrics to MLflow
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    
    # Save model
    model_path = save_model(model, feature_cols, metrics, args.model_output)
    
    # Log model to MLflow
    mlflow.xgboost.log_model(model, "model")
    
    # End MLflow run
    mlflow.end_run()
    
    print("\n✅ Training pipeline completed successfully!")


if __name__ == "__main__":
    main()