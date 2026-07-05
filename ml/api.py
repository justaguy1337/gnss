"""
FastAPI Server for GNSS Error Prediction
==========================================
REST API bridging the Python ML backend with the React frontend.

Endpoints:
  POST /api/upload       — Upload CSV data
  POST /api/train        — Launch training pipeline
  GET  /api/predict/{h}  — Predictions for horizon h
  GET  /api/predict/all  — All predictions
  GET  /api/models       — Model info and weights
  GET  /api/evaluation   — Evaluation metrics
  GET  /api/status       — Pipeline status
"""

import os
import sys
import json
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import HORIZONS, RESULTS_DIR, DATA_DIR, CHECKPOINT_DIR
from data.data_loader import GNSSDataset
from pipeline.train import GNSSEnsemble

app = FastAPI(
    title="GNSS ErrorNet API",
    description="AI/ML Satellite Error Prediction Backend",
    version="1.0.0"
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
pipeline_state = {
    "status": "idle",            # idle, training, trained, error
    "progress": 0,
    "message": "",
    "dataset": None,
    "ensemble": None,
    "predictions": None,
    "evaluation": None,
}


@app.get("/api/status")
async def get_status():
    """Get current pipeline status."""
    return {
        "status": pipeline_state["status"],
        "progress": pipeline_state["progress"],
        "message": pipeline_state["message"],
        "has_data": pipeline_state["dataset"] is not None,
        "has_model": pipeline_state["ensemble"] is not None,
        "has_predictions": pipeline_state["predictions"] is not None,
    }


@app.post("/api/upload")
async def upload_data(file: UploadFile = File(...)):
    """Upload CSV data file."""
    try:
        # Save uploaded file
        os.makedirs(DATA_DIR, exist_ok=True)
        filepath = os.path.join(DATA_DIR, file.filename)
        content = await file.read()
        with open(filepath, "wb") as f:
            f.write(content)

        # Load dataset
        dataset = GNSSDataset(filepath)
        pipeline_state["dataset"] = dataset

        return {
            "status": "success",
            "message": f"Uploaded {file.filename}",
            "summary": dataset.summary()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/train")
async def start_training(
    background_tasks: BackgroundTasks,
    satellite: Optional[str] = None,
    error_col: Optional[str] = None,
    epochs: int = 50,
    quick: bool = False
):
    """Start training pipeline (runs in background)."""
    if pipeline_state["status"] == "training":
        raise HTTPException(status_code=409, detail="Training already in progress")

    # Load dataset if not already loaded
    if pipeline_state["dataset"] is None:
        try:
            pipeline_state["dataset"] = GNSSDataset()
        except FileNotFoundError:
            raise HTTPException(
                status_code=400,
                detail="No data available. Upload data or generate synthetic data first."
            )

    dataset = pipeline_state["dataset"]
    sat_id = satellite or dataset.satellite_ids[0]

    def train_task():
        try:
            pipeline_state["status"] = "training"
            pipeline_state["progress"] = 0
            pipeline_state["message"] = f"Training {sat_id}..."

            train_series, test_series, scaler = dataset.get_satellite_data(
                sat_id, error_col, normalize=False
            )

            ensemble = GNSSEnsemble()
            results = ensemble.fit(
                train_series,
                epochs=epochs,
                quick=quick,
                verbose=True
            )

            pipeline_state["ensemble"] = ensemble
            pipeline_state["progress"] = 80
            pipeline_state["message"] = "Generating predictions..."

            # Generate predictions
            from pipeline.predict import predict_day8
            predictions = predict_day8(ensemble, train_series, test_series)
            pipeline_state["predictions"] = predictions

            pipeline_state["progress"] = 90
            pipeline_state["message"] = "Evaluating..."

            # Evaluate
            from pipeline.evaluate import full_evaluation
            evaluation = full_evaluation(predictions)
            pipeline_state["evaluation"] = evaluation

            # Save
            ensemble.save()

            pipeline_state["status"] = "trained"
            pipeline_state["progress"] = 100
            pipeline_state["message"] = "Training complete"

        except Exception as e:
            pipeline_state["status"] = "error"
            pipeline_state["message"] = str(e)
            import traceback
            traceback.print_exc()

    background_tasks.add_task(train_task)

    return {
        "status": "started",
        "message": f"Training started for {sat_id}",
        "satellite": sat_id,
    }


@app.get("/api/predict/all")
async def get_all_predictions():
    """Get predictions for all horizons."""
    if pipeline_state["predictions"] is None:
        raise HTTPException(status_code=404, detail="No predictions available. Train first.")

    return pipeline_state["predictions"]


@app.get("/api/predict/{horizon}")
async def get_predictions(horizon: int):
    """Get predictions for a specific horizon."""
    if pipeline_state["predictions"] is None:
        raise HTTPException(status_code=404, detail="No predictions available")

    if horizon not in pipeline_state["predictions"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid horizon {horizon}. Valid: {HORIZONS}"
        )

    return pipeline_state["predictions"][horizon]


@app.get("/api/models")
async def get_models():
    """Get model information and Ridge stacker weights."""
    ensemble = pipeline_state["ensemble"]
    if ensemble is None:
        raise HTTPException(status_code=404, detail="No trained model available")

    models_info = {
        "base_models": ["LSTM-GRU", "Transformer", "XGBoost"],
        "meta_learner": "Ridge Regression (per-horizon)",
        "residual_model": "Gaussian Process (Matérn 2.5 + Periodic)",
        "horizons": {
            h: ensemble.stackers[h].get_weights_info()
            for h in HORIZONS if h in ensemble.stackers
        },
    }

    return models_info


@app.get("/api/evaluation")
async def get_evaluation():
    """Get evaluation metrics and normality test results."""
    if pipeline_state["evaluation"] is None:
        raise HTTPException(status_code=404, detail="No evaluation available")

    return pipeline_state["evaluation"]


@app.get("/api/evaluation/distribution")
async def get_distribution_data():
    """Get residual distribution data for plotting."""
    if pipeline_state["evaluation"] is None:
        raise HTTPException(status_code=404, detail="No evaluation available")

    dist_data = {}
    for h, metrics in pipeline_state["evaluation"].items():
        dist_data[h] = {
            "qq_data": metrics.get("qq_data"),
            "histogram_data": metrics.get("histogram_data"),
            "normality": {
                "shapiro_wilk": metrics.get("shapiro_wilk"),
                "anderson_darling": metrics.get("anderson_darling"),
            }
        }

    return dist_data


@app.get("/api/data/summary")
async def get_data_summary():
    """Get dataset summary statistics."""
    if pipeline_state["dataset"] is None:
        raise HTTPException(status_code=404, detail="No data loaded")

    return pipeline_state["dataset"].summary()


@app.post("/api/generate-synthetic")
async def generate_synthetic(n_meo: int = 4, n_geo: int = 2):
    """Generate synthetic data for testing."""
    try:
        from data.generate_synthetic import generate_full_dataset
        df = generate_full_dataset(n_meo=n_meo, n_geo=n_geo)

        # Load as dataset
        pipeline_state["dataset"] = GNSSDataset()

        return {
            "status": "success",
            "message": f"Generated synthetic data: {len(df)} rows",
            "summary": pipeline_state["dataset"].summary()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
