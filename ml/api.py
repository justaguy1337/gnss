"""
FastAPI Server for GNSS Error Prediction
==========================================
REST API bridging the Python ML backend with the React frontend.

Startup behaviour
-----------------
On every launch the server:
  1. Loads all ISRO training data from DATASET_DIR (d:/github/gnss/dataset/)
     automatically — no user action required.
  2. Checks for existing checkpoints (CHECKPOINT_DIR).
     - If valid checkpoints exist -> loads them (fast startup, ~seconds).
     - If no checkpoints found   -> trains the full ensemble (~minutes).
  3. Saves checkpoints after training.
  4. Only then starts accepting requests.

Upload policy
-------------
  POST /api/upload  accepts TEST data only.
  Uploading a file whose name contains 'train' (case-insensitive) is rejected
  with HTTP 400. Training data is always sourced from DATASET_DIR.

No synthetic data
-----------------
  The /api/generate-synthetic endpoint has been removed.
  No synthetic fallback exists anywhere in the stack.

Endpoints
---------
  POST /api/upload         — Upload test CSV (test data only)
  POST /api/train          — Retrain on ISRO data (background)
  GET  /api/predict/all    — All predictions
  GET  /api/predict/{h}    — Predictions for horizon h
  GET  /api/models         — Model info and stacker weights
  GET  /api/evaluation     — Evaluation metrics
  GET  /api/status         — Pipeline status
  GET  /api/data/summary   — Dataset summary
"""

import os
import sys
import json
import time
import numpy as np
from typing import Optional
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import HORIZONS, RESULTS_DIR, DATA_DIR, CHECKPOINT_DIR, DATASET_DIR
from data.data_loader import GNSSDataset, validate_schema
from pipeline.train import GNSSEnsemble, train_all_satellites

# =============================================================================
# Lifespan — load data and train/restore before accepting requests
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    startup_event()   # runs synchronously before server accepts connections
    yield
    # (shutdown hook — nothing to clean up)


app = FastAPI(
    title="GNSS ErrorNet API",
    description="AI/ML Satellite Error Prediction Backend -- real ISRO data only",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────
pipeline_state = {
    "status":    "initializing",   # initializing | idle | training | trained | error
    "progress":  0,
    "message":   "Server starting — loading ISRO dataset and checking checkpoints...",
    "dataset":   None,             # GNSSDataset (training data, loaded at startup)
    "ensembles": {},               # sat_id -> GNSSEnsemble (trained per satellite)
    "ensemble":  None,             # primary satellite ensemble (for backward compat)
    "predictions": None,
    "evaluation":  None,
    "trained_satellites": [],
    "startup_time": None,
}


# =============================================================================
# Startup — load data and train/restore before accepting requests
# =============================================================================

def _checkpoints_exist() -> bool:
    """Return True if checkpoints for all configured horizons are on disk."""
    if not os.path.isdir(CHECKPOINT_DIR):
        return False
    for h in HORIZONS:
        if not os.path.exists(os.path.join(CHECKPOINT_DIR, f"lstm_gru_h{h}.pt")):
            return False
        if not os.path.exists(os.path.join(CHECKPOINT_DIR, f"stacker_h{h}.pkl")):
            return False
    return True


def _run_predictions_and_evaluation(
    dataset: GNSSDataset,
    ensembles: dict,
) -> None:
    """
    Generate predictions and evaluation for the primary satellite (GEO preferred).
    Stores results in pipeline_state and saves JSON files to RESULTS_DIR.
    """
    from pipeline.predict import predict_day8
    from pipeline.evaluate import full_evaluation

    primary_id = "GEO" if "GEO" in ensembles else list(ensembles.keys())[0]
    ensemble   = ensembles[primary_id]
    error_col  = dataset.get_default_error_col()

    train_series, test_series, _ = dataset.get_satellite_data(
        primary_id, error_col, normalize=False
    )

    predictions = predict_day8(ensemble, train_series, test_series, verbose=False)
    pipeline_state["predictions"] = predictions

    # Save predictions JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pred_path = os.path.join(RESULTS_DIR, "day8_predictions.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2, default=str)

    evaluation = full_evaluation(predictions)
    pipeline_state["evaluation"] = evaluation

    # Save evaluation JSON
    eval_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    with open(eval_path, "w") as f:
        json.dump(evaluation, f, indent=2, default=str)


def startup_event():
    """
    Runs synchronously before the server accepts any connections.

    Steps:
      1. Load ISRO dataset (all training + test files from DATASET_DIR)
      2a. If checkpoints exist -> load them (fast path, ~seconds)
      2b. If no checkpoints   -> train all satellites and save (slow path, ~minutes)
      3. Generate initial predictions and evaluation
    """
    t0 = time.time()
    print("\n" + "=" * 60)
    print("  GNSS ErrorNet API -- Startup")
    print("=" * 60)

    # 1. Load dataset
    try:
        pipeline_state["message"] = "Loading ISRO dataset..."
        dataset = GNSSDataset()
        pipeline_state["dataset"] = dataset
        print(f"  Dataset loaded: {dataset.satellite_ids}")
    except Exception as e:
        pipeline_state["status"]  = "error"
        pipeline_state["message"] = f"FATAL: Could not load ISRO dataset: {e}"
        print(f"\n  ERROR: {e}")
        return

    error_col = dataset.get_default_error_col()

    # 2. Train or restore checkpoints
    if _checkpoints_exist():
        # Fast path — load existing checkpoints
        print(f"  Checkpoints found in '{CHECKPOINT_DIR}' -- loading...")
        pipeline_state["message"] = "Loading saved model checkpoints..."
        ensembles = {}
        for sat_id in dataset.satellite_ids:
            ensemble = GNSSEnsemble()
            ensemble.load(CHECKPOINT_DIR)
            ensembles[sat_id] = ensemble
        print("  Checkpoints loaded.")
    else:
        # Slow path — train from scratch on all satellites
        print(f"  No checkpoints found — training on all satellites...")
        print(f"  (This runs once; subsequent restarts will use checkpoints)")
        pipeline_state["status"]  = "training"
        pipeline_state["message"] = "Training GNSS ensemble on ISRO dataset..."

        try:
            ensembles = train_all_satellites(
                dataset, error_col, verbose=True
            )
            # Save primary satellite checkpoints
            primary_id = "GEO" if "GEO" in ensembles else list(ensembles.keys())[0]
            ensembles[primary_id].save(CHECKPOINT_DIR)
            print(f"  Checkpoints saved to '{CHECKPOINT_DIR}'")
        except Exception as e:
            pipeline_state["status"]  = "error"
            pipeline_state["message"] = f"Training failed: {e}"
            import traceback
            traceback.print_exc()
            return

    # Store ensembles
    pipeline_state["ensembles"]           = ensembles
    pipeline_state["ensemble"]            = ensembles.get(
        "GEO", next(iter(ensembles.values()))
    )
    pipeline_state["trained_satellites"]  = list(ensembles.keys())

    # 3. Generate initial predictions and evaluation
    try:
        pipeline_state["message"] = "Generating initial predictions..."
        _run_predictions_and_evaluation(dataset, ensembles)
    except Exception as e:
        print(f"  WARNING: Prediction/evaluation failed: {e}")
        import traceback
        traceback.print_exc()

    elapsed = time.time() - t0
    pipeline_state["status"]       = "trained"
    pipeline_state["progress"]     = 100
    pipeline_state["message"]      = (
        f"Ready -- {len(ensembles)} satellite(s) trained "
        f"[{', '.join(ensembles.keys())}] in {elapsed:.1f}s"
    )
    pipeline_state["startup_time"] = elapsed

    print(f"\n  Startup complete in {elapsed:.1f}s — API ready.")
    print("=" * 60 + "\n")


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/api/status")
async def get_status():
    """Get current pipeline status."""
    return {
        "status":       pipeline_state["status"],
        "progress":     pipeline_state["progress"],
        "message":      pipeline_state["message"],
        "has_data":     pipeline_state["dataset"] is not None,
        "has_model":    bool(pipeline_state["ensembles"]),
        "has_predictions": pipeline_state["predictions"] is not None,
        "trained_satellites": pipeline_state["trained_satellites"],
        "startup_time": pipeline_state["startup_time"],
    }


@app.post("/api/upload")
async def upload_test_data(file: UploadFile = File(...)):
    """
    Upload a test CSV file.

    Restrictions
    ------------
    - Test CSVs only. Files whose name contains 'train' (case-insensitive)
      are rejected with HTTP 400. Training data is loaded automatically from
      DATASET_DIR at startup and cannot be replaced via this endpoint.
    - Schema is validated against the loaded training data before acceptance.
      Any column mismatch returns HTTP 422 with a precise diff.

    Expected CSV schema
    -------------------
    utc_time, x_error (m), y_error (m), z_error (m), satclockerror (m)
    """
    fname_upper = file.filename.upper()

    # Reject training files
    if "TRAIN" in fname_upper:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Upload rejected: '{file.filename}' appears to be a training file. "
                f"Training data is loaded automatically from the ISRO dataset folder "
                f"('{DATASET_DIR}') at server startup and cannot be replaced here. "
                f"Only test dataset files may be uploaded."
            )
        )

    if pipeline_state["dataset"] is None:
        raise HTTPException(
            status_code=503,
            detail="Server is still initializing. Retry in a few seconds."
        )

    # Save to DATA_DIR
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, file.filename)
    content  = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)

    # Load + schema-validate via GNSSDataset
    try:
        dataset = pipeline_state["dataset"]
        sat_id  = dataset.load_test_csv(filepath)
    except ValueError as e:
        # Schema mismatch — return precise column diff
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Regenerate predictions with the new test data
    try:
        _run_predictions_and_evaluation(
            pipeline_state["dataset"],
            pipeline_state["ensembles"]
        )
    except Exception as e:
        print(f"  WARNING: prediction re-run failed after upload: {e}")

    return {
        "status":    "success",
        "message":   f"Test data uploaded and validated: {file.filename}",
        "satellite": sat_id,
        "summary":   pipeline_state["dataset"].summary(),
    }


@app.post("/api/train")
async def retrain(
    background_tasks: BackgroundTasks,
    satellite: Optional[str] = None,
    error_col: Optional[str] = None,
    epochs: int = 50,
    quick: bool = False,
):
    """
    Retrain the ensemble on the ISRO dataset (runs in background).

    Training data always comes from DATASET_DIR — the dataset loaded at startup.
    Use this to re-run training with different epoch counts or after updates
    to the dataset files on disk.
    """
    if pipeline_state["status"] == "training":
        raise HTTPException(status_code=409, detail="Training already in progress.")

    if pipeline_state["dataset"] is None:
        raise HTTPException(status_code=503, detail="Dataset not yet loaded.")

    dataset   = pipeline_state["dataset"]
    sat_id    = satellite or dataset.satellite_ids[0]
    error_col = error_col or dataset.get_default_error_col()

    def train_task():
        try:
            pipeline_state["status"]   = "training"
            pipeline_state["progress"] = 0
            pipeline_state["message"]  = f"Retraining {sat_id}..."

            train_series, test_series, _ = dataset.get_satellite_data(
                sat_id, error_col, normalize=False
            )

            ensemble = GNSSEnsemble()
            ensemble.fit(train_series, epochs=epochs, quick=quick, verbose=True)
            ensemble.save(CHECKPOINT_DIR)

            pipeline_state["ensembles"][sat_id] = ensemble
            pipeline_state["ensemble"]          = ensemble
            pipeline_state["progress"]          = 80
            pipeline_state["message"]           = "Generating predictions..."

            _run_predictions_and_evaluation(dataset, pipeline_state["ensembles"])

            pipeline_state["status"]   = "trained"
            pipeline_state["progress"] = 100
            pipeline_state["message"]  = f"Retraining complete for {sat_id}."

        except Exception as e:
            pipeline_state["status"]  = "error"
            pipeline_state["message"] = str(e)
            import traceback
            traceback.print_exc()

    background_tasks.add_task(train_task)
    return {
        "status":    "started",
        "message":   f"Retraining started for satellite '{sat_id}'",
        "satellite": sat_id,
    }


@app.get("/api/predict/all")
async def get_all_predictions():
    """Get predictions for all horizons."""
    if pipeline_state["predictions"] is None:
        raise HTTPException(status_code=404, detail="No predictions available. Check /api/status.")
    return pipeline_state["predictions"]


@app.get("/api/predict/{horizon}")
async def get_predictions(horizon: int):
    """Get predictions for a specific horizon (step count, not minutes)."""
    if pipeline_state["predictions"] is None:
        raise HTTPException(status_code=404, detail="No predictions available.")
    if horizon not in pipeline_state["predictions"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid horizon {horizon}. Valid horizons: {HORIZONS}"
        )
    return pipeline_state["predictions"][horizon]


@app.get("/api/models")
async def get_models():
    """Get model information and Ridge stacker weights per horizon."""
    if not pipeline_state["ensembles"]:
        raise HTTPException(status_code=404, detail="No trained models available.")

    ensemble = pipeline_state["ensemble"]
    return {
        "base_models":     ["LSTM-GRU", "Transformer", "XGBoost"],
        "meta_learner":    "Ridge Regression (per-horizon)",
        "residual_model":  "Gaussian Process (Matern 2.5 + Periodic)",
        "trained_satellites": pipeline_state["trained_satellites"],
        "horizons": {
            h: ensemble.stackers[h].get_weights_info()
            for h in HORIZONS if h in ensemble.stackers
        },
    }


@app.get("/api/evaluation")
async def get_evaluation():
    """Get evaluation metrics and normality test results."""
    if pipeline_state["evaluation"] is None:
        raise HTTPException(status_code=404, detail="No evaluation available.")
    return pipeline_state["evaluation"]


@app.get("/api/evaluation/distribution")
async def get_distribution_data():
    """Get residual distribution data for plotting (Q-Q, histogram)."""
    if pipeline_state["evaluation"] is None:
        raise HTTPException(status_code=404, detail="No evaluation available.")

    return {
        h: {
            "qq_data":        metrics.get("qq_data"),
            "histogram_data": metrics.get("histogram_data"),
            "normality": {
                "shapiro_wilk":    metrics.get("shapiro_wilk"),
                "anderson_darling": metrics.get("anderson_darling"),
            },
        }
        for h, metrics in pipeline_state["evaluation"].items()
    }


@app.get("/api/data/summary")
async def get_data_summary():
    """Get dataset summary statistics."""
    if pipeline_state["dataset"] is None:
        raise HTTPException(status_code=503, detail="Dataset not yet loaded.")
    return pipeline_state["dataset"].summary()


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
