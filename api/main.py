"""
Energy Consumption Prediction API
FastAPI service for serving XGBoost model predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import os
import asyncpg
from contextlib import asynccontextmanager

# Global variables
model = None
feature_cols = None
metadata = None
db_pool = None


class PredictionRequest(BaseModel):
    hours_ahead: int = Field(default=24, ge=1, le=168, description="Hours ahead to predict (1-168)")
    current_power: Optional[float] = Field(default=None, description="Current power consumption in kW")
    voltage: Optional[float] = Field(default=240.0, description="Voltage")
    intensity: Optional[float] = Field(default=5.0, description="Global intensity")


class PredictionResponse(BaseModel):
    predictions: List[dict]
    model_version: str
    prediction_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str]
    database_connected: bool
    timestamp: str


async def init_db():
    """Initialize database connection pool"""
    global db_pool
    
    db_host = os.getenv("DB_HOST")
    db_name = os.getenv("DB_NAME", "energy_predictions")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    
    if all([db_host, db_user, db_password]):
        try:
            db_pool = await asyncpg.create_pool(
                host=db_host,
                database=db_name,
                user=db_user,
                password=db_password,
                port=5432,
                min_size=1,
                max_size=10
            )
            print("✅ Database connection pool created")
        except Exception as e:
            print(f"⚠️ Database connection failed: {e}")
            db_pool = None
    else:
        print("⚠️ Database credentials not provided, running without DB")


async def close_db():
    """Close database connection pool"""
    global db_pool
    if db_pool:
        await db_pool.close()
        print("Database connection pool closed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    load_model()
    await init_db()
    yield
    # Shutdown
    await close_db()


# Initialize FastAPI app
app = FastAPI(
    title="Energy Consumption Prediction API",
    description="Predict household energy consumption using XGBoost",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model():
    """Load the trained model and metadata"""
    global model, feature_cols, metadata
    
    model_dir = Path("./model")
    
    try:
        # Load model
        model_path = model_dir / "model.pkl"
        if model_path.exists():
            model = joblib.load(model_path)
            print(f"✅ Model loaded from {model_path}")
        else:
            print(f"⚠️ Model not found at {model_path}")
            return
        
        # Load features
        features_path = model_dir / "features.json"
        if features_path.exists():
            with open(features_path, 'r') as f:
                feature_cols = json.load(f)['features']
            print(f"✅ Features loaded: {len(feature_cols)} features")
        
        # Load metadata
        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"✅ Metadata loaded")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None


def create_features(hours_ahead: int, current_power: float = None, 
                   voltage: float = 240.0, intensity: float = 5.0) -> pd.DataFrame:
    """Create feature DataFrame for prediction"""
    
    # Current time
    now = datetime.now()
    
    # Generate timestamps for predictions
    timestamps = [now + timedelta(hours=i) for i in range(1, hours_ahead + 1)]
    
    # Create base features
    features_list = []
    
    for ts in timestamps:
        features = {
            'Voltage': voltage,
            'Global_intensity': intensity,
            'Sub_metering_1': 0.0,
            'Sub_metering_2': 0.0,
            'Sub_metering_3': 0.0,
            'hour': ts.hour,
            'day_of_week': ts.weekday(),
            'day_of_month': ts.day,
            'month': ts.month,
            'quarter': (ts.month - 1) // 3 + 1,
            'is_weekend': 1 if ts.weekday() >= 5 else 0,
            'hour_sin': np.sin(2 * np.pi * ts.hour / 24),
            'hour_cos': np.cos(2 * np.pi * ts.hour / 24),
            'month_sin': np.sin(2 * np.pi * ts.month / 12),
            'month_cos': np.cos(2 * np.pi * ts.month / 12),
        }
        
        # Add lag features (use current_power or historical average)
        power_value = current_power if current_power else 1.2  # historical average
        features['power_lag_1h'] = power_value
        features['power_lag_24h'] = power_value
        features['power_lag_7d'] = power_value
        features['power_roll_mean_24h'] = power_value
        features['power_roll_std_24h'] = 0.3
        features['power_roll_min_24h'] = power_value * 0.7
        features['power_roll_max_24h'] = power_value * 1.3
        
        features_list.append(features)
    
    # Create DataFrame
    df = pd.DataFrame(features_list)
    
    # Ensure column order matches training
    if feature_cols:
        df = df[feature_cols]
    
    return df, timestamps


async def store_prediction(timestamp: datetime, hours_ahead: int, 
                          predicted_power: float, model_version: str):
    """Store prediction in database"""
    if not db_pool:
        return
    
    try:
        async with db_pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO predictions (timestamp, hours_ahead, predicted_power, model_version)
                VALUES ($1, $2, $3, $4)
            ''', timestamp, hours_ahead, predicted_power, model_version)
    except Exception as e:
        print(f"Error storing prediction: {e}")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Energy Consumption Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model-info",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model else "degraded",
        model_loaded=model is not None,
        model_version=metadata.get('trained_at', 'unknown') if metadata else None,
        database_connected=db_pool is not None,
        timestamp=datetime.now().isoformat()
    )


@app.get("/model-info", response_model=dict)
async def model_info():
    """Get model information"""
    if not model or not metadata:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": metadata.get('model_type', 'Unknown'),
        "trained_at": metadata.get('trained_at', 'Unknown'),
        "n_features": metadata.get('n_features', 0),
        "metrics": metadata.get('metrics', {}),
        "features": feature_cols[:10] if feature_cols else []  # First 10 features
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make energy consumption predictions"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = datetime.now()
    
    try:
        # Create features
        X, timestamps = create_features(
            request.hours_ahead,
            request.current_power,
            request.voltage,
            request.intensity
        )
        
        # Make predictions
        predictions = model.predict(X)
        
        # Format response
        results = []
        for ts, pred in zip(timestamps, predictions):
            result = {
                "timestamp": ts.isoformat(),
                "predicted_power": round(float(pred), 4)
            }
            results.append(result)
            
            # Store in database (async)
            if db_pool:
                await store_prediction(
                    ts, request.hours_ahead,
                    float(pred),
                    metadata.get('trained_at', 'unknown') if metadata else 'unknown'
                )
        
        # Calculate prediction time
        prediction_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return PredictionResponse(
            predictions=results,
            model_version=metadata.get('trained_at', 'unknown') if metadata else 'unknown',
            prediction_time_ms=round(prediction_time, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/predictions/recent", response_model=dict)
async def get_recent_predictions(limit: int = 10):
    """Get recent predictions from database"""
    
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT timestamp, hours_ahead, predicted_power, 
                       actual_power, error, model_version, created_at
                FROM predictions
                ORDER BY created_at DESC
                LIMIT $1
            ''', limit)
            
            results = [dict(row) for row in rows]
            
            return {
                "count": len(results),
                "predictions": results
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")


@app.get("/predictions/stats", response_model=dict)
async def get_prediction_stats():
    """Get prediction statistics"""
    
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        async with db_pool.acquire() as conn:
            # Get total predictions
            total = await conn.fetchval('SELECT COUNT(*) FROM predictions')
            
            # Get average error (where actual is available)
            avg_error = await conn.fetchval('''
                SELECT AVG(ABS(error)) 
                FROM predictions 
                WHERE actual_power IS NOT NULL
            ''')
            
            # Get predictions by hour
            hourly = await conn.fetch('''
                SELECT EXTRACT(HOUR FROM timestamp) as hour, 
                       AVG(predicted_power) as avg_power,
                       COUNT(*) as count
                FROM predictions
                GROUP BY hour
                ORDER BY hour
            ''')
            
            return {
                "total_predictions": total,
                "average_absolute_error": float(avg_error) if avg_error else None,
                "hourly_patterns": [dict(row) for row in hourly]
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistics query failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)