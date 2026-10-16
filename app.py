from pathlib import Path
import json
import joblib
import pandas as pd

from flask import Flask, render_template, request

from ml.data_loader import build_dataset
from ml.model import load_models

from models.load import load_data
from models.eda import run_full_eda
from models.regression import (
    run_full_regression,
    predict_single,
    get_plant_ids
)
from unsupervised import run_all as run_unsupervised

from models.ensemble import (
    run_full_ensemble,
    predict_single as predict_single_ensemble,
    get_plant_ids as get_plant_ids_ensemble
)


# ============================================================
# FLASK APPLICATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)


# ============================================================
# FILE PATHS
# ============================================================

DATA_PATH = BASE_DIR / "data" / "processed" / "solar_sense_processed.csv"

METRICS_PATH = BASE_DIR / "models" / "metrics.json"

FAULT_MODEL_PATH = BASE_DIR / "models" / "fault_model.joblib"

GEN_MODEL_PATH = BASE_DIR / "models" / "generation_model.joblib"

DATASET_NAME = "Solar Power Plant Generation & Weather Data"


# ============================================================
# LOAD PROCESSED DATASET
# ============================================================

def load_processed():
    """
    Load the processed SolarSense dataset.

    If the processed dataset does not exist,
    build_dataset() will create it.
    """

    if not DATA_PATH.exists():
        build_dataset(save=True)

    return pd.read_csv(
        DATA_PATH,
        parse_dates=["DATE_TIME"]
    )


# ============================================================
# GET DROPDOWN OPTIONS
# ============================================================

def options():
    """
    Get available plant IDs and source keys
    for the prediction page.
    """

    df = load_processed()

    plants = sorted(
        df["PLANT_ID"]
        .astype(int)
        .unique()
    )

    sources = sorted(
        df["SOURCE_KEY"]
        .astype(str)
        .unique()
    )

    return plants, sources


# ============================================================
# HOME / DASHBOARD
# ============================================================

@app.route("/")
def index():

    df = load_processed()

    # Load model metrics if available
    if METRICS_PATH.exists():

        metrics = json.loads(
            METRICS_PATH.read_text(
                encoding="utf-8"
            )
        )

    else:

        metrics = {}


    # Dashboard summary
    summary = {

        "rows": len(df),

        "columns": len(df.columns),

        "plants": df["PLANT_ID"].nunique(),

        "inverters": df["SOURCE_KEY"].nunique(),

        "faults": int(
            df["FAULT_FLAG"].sum()
        ),

        "fault_rate": round(
            float(
                df["FAULT_FLAG"].mean() * 100
            ),
            2
        ),

        "total_ac": round(
            float(
                df["AC_POWER"].sum()
            ),
            2
        ),

        "max_ac": round(
            float(
                df["AC_POWER"].max()
            ),
            2
        ),

    }


    return render_template(
        "index.html",
        summary=summary,
        metrics=metrics
    )


# ============================================================
# DATASET PAGE
# ============================================================

@app.route("/data")
def data_page():

    # Load processed dataset
    df = load_processed()


    # --------------------------------------------------------
    # Get first 12 rows
    # --------------------------------------------------------

    preview_df = df.head(12)


    # --------------------------------------------------------
    # Convert DataFrame to HTML table
    # --------------------------------------------------------

    preview = preview_df.to_html(
        classes="data-table",
        index=False,
        border=0
    )


    # --------------------------------------------------------
    # Get column names
    # --------------------------------------------------------

    columns = list(df.columns)


    # --------------------------------------------------------
    # Send data to data.html
    # --------------------------------------------------------

    return render_template(
        "data.html",

        preview=preview,

        columns=columns,

        rows=len(df),

        columns_count=len(columns)
    )



@app.route("/eda")
def eda_page():

    try:

        df = load_data()

        result = run_full_eda(df)


        return render_template(
            "eda.html",

            dataset_name=DATASET_NAME,

            error=None,

            **result
        )


    except Exception as exc:

        return render_template(
            "eda.html",

            dataset_name=DATASET_NAME,

            error=str(exc),

            info={},

            stats={},

            missing=[],

            missing_total=0,

            duplicates=0,

            uniques={},

            cat_frequencies={},

            outliers=[],

            correlation_matrix={},

            target={},

            plots=[],

            findings=[]
        )


# ============================================================
# REGRESSION PAGE
# ============================================================

@app.route("/regression")
def regression_page():

    try:

        df = load_data()

        result = run_full_regression(df)


        return render_template(
            "regression.html",

            dataset_name=DATASET_NAME,

            plant_ids=get_plant_ids(df),

            error=None,

            **result
        )


    except Exception as exc:

        return render_template(
            "regression.html",

            dataset_name=DATASET_NAME,

            error=str(exc),

            target="AC_POWER",

            features=[],

            train_size=0,

            test_size=0,

            models={},

            plots=[],

            best_model=None,

            feature_importances=None,

            findings=[]
        )


# ============================================================
# ENSEMBLE PAGE
# ============================================================

@app.route("/ensemble")
def ensemble_page():

    try:

        df = load_data()

        result = run_full_ensemble(df)


        return render_template(
            "ensemble.html",

            dataset_name=DATASET_NAME,

            plant_ids=get_plant_ids_ensemble(df),

            error=None,

            **result
        )


    except Exception as exc:

        return render_template(
            "ensemble.html",

            dataset_name=DATASET_NAME,

            error=str(exc),

            target="AC_POWER",

            features=[],

            train_size=0,

            test_size=0,

            models={},

            plots=[],

            best_model=None,

            feature_importances=None,

            xgboost_available=False,

            findings=[]
        )


# ============================================================
# PREDICTION PAGE
# ============================================================

@app.route("/predict", methods=["GET", "POST"])
def predict_page():

    # Get dropdown options
    plants, sources = options()

    result = None

    error = None


    # ========================================================
    # HANDLE POST REQUEST
    # ========================================================

    if request.method == "POST":

        try:

            # ------------------------------------------------
            # Convert date and time
            # ------------------------------------------------

            dt = pd.to_datetime(
                request.form["date_time"]
            )


            # ------------------------------------------------
            # Create input row
            # ------------------------------------------------

            row = {

                "DC_POWER": float(
                    request.form["dc_power"]
                ),

                "DAILY_YIELD": float(
                    request.form["daily_yield"]
                ),

                "IRRADIATION": float(
                    request.form["irradiation"]
                ),

                "AMBIENT_TEMPERATURE": float(
                    request.form["ambient_temperature"]
                ),

                "MODULE_TEMPERATURE": float(
                    request.form["module_temperature"]
                ),

                "HOUR": int(
                    dt.hour
                ),

                "MINUTE": int(
                    dt.minute
                ),

                "DAY_OF_WEEK": int(
                    dt.dayofweek
                ),

                "MONTH": int(
                    dt.month
                ),

                "DAY_OF_YEAR": int(
                    dt.dayofyear
                ),

                "PLANT_ID": int(
                    request.form["plant_id"]
                ),

                "SOURCE_KEY": request.form[
                    "source_key"
                ],
            }


            # ------------------------------------------------
            # Convert input into DataFrame
            # ------------------------------------------------

            x = pd.DataFrame([row])


            # ------------------------------------------------
            # Load trained models
            # ------------------------------------------------

            fault_model, gen_model = load_models()


            # ------------------------------------------------
            # Fault prediction
            # ------------------------------------------------

            fault = int(
                fault_model.predict(x)[0]
            )


            # ------------------------------------------------
            # Generation model features
            # ------------------------------------------------

            gen_features = [

                "IRRADIATION",

                "AMBIENT_TEMPERATURE",

                "MODULE_TEMPERATURE",

                "HOUR",

                "MINUTE",

                "DAY_OF_WEEK",

                "MONTH",

                "DAY_OF_YEAR",

                "PLANT_ID",

                "SOURCE_KEY"

            ]


            # ------------------------------------------------
            # Generation prediction
            # ------------------------------------------------

            gen = float(
                gen_model.predict(
                    x[gen_features]
                )[0]
            )


            # ------------------------------------------------
            # Fault probability
            # ------------------------------------------------

            probability = float(
                fault_model.predict_proba(x)[0][1]
            )


            # ------------------------------------------------
            # Prediction result
            # ------------------------------------------------

            result = {

                "fault": fault,

                "fault_probability": round(
                    probability * 100,
                    2
                ),

                "generation": round(
                    max(gen, 0),
                    2
                ),

            }


        except Exception as exc:

            error = str(exc)


    # ========================================================
    # RENDER PREDICTION PAGE
    # ========================================================

    return render_template(
        "prediction.html",

        plants=plants,

        sources=sources,

        result=result,

        error=error
    )


@app.route(
    "/predict_regression",
    methods=["POST"]
)
def predict_regression_api():

    try:

        # Get JSON request
        data = request.get_json(
            force=True
        )


        # Make prediction
        pred = predict_single(

            ambient_temp=data[
                "ambient_temperature"
            ],

            module_temp=data[
                "module_temperature"
            ],

            irradiation=data[
                "irradiation"
            ],

            plant_id=data[
                "plant_id"
            ],

        )


        return {

            "status": "ok",

            "prediction": pred

        }


    except Exception as exc:

        return {

            "status": "error",

            "message": str(exc)

        }, 400



@app.route(
    "/predict_ensemble",
    methods=["POST"]
)
def predict_ensemble_api():

    try:

        # Get JSON request
        data = request.get_json(
            force=True
        )


        # Make ensemble prediction
        pred = predict_single_ensemble(

            ambient_temp=data[
                "ambient_temperature"
            ],

            module_temp=data[
                "module_temperature"
            ],

            irradiation=data[
                "irradiation"
            ],

            plant_id=data[
                "plant_id"
            ],

        )


        return {

            "status": "ok",

            "prediction": pred

        }


    except Exception as exc:

        return {

            "status": "error",

            "message": str(exc)

        }, 400




# ============================================================
# UNSUPERVISED LEARNING PAGE
# ============================================================

@app.route("/unsupervised")
def unsupervised_page():
    try:
        df = load_processed()
        result = run_unsupervised(df)

        return render_template(
            "unsupervised.html",
            dataset_name=DATASET_NAME,
            error=None,
            **result
        )

    except Exception as exc:
        return render_template(
            "unsupervised.html",
            dataset_name=DATASET_NAME,
            error=str(exc),
            features=[],
            rows=0,
            kmeans={"clusters": 0, "silhouette": None, "plot": None, "summary": []},
            hierarchical={"clusters": 0, "sample_size": 0, "silhouette": None, "plot": None},
            dbscan={"clusters": 0, "sample_size": 0, "noise": 0, "noise_percent": 0, "plot": None},
            pca={"variance": [], "total_variance": 0, "plot": None},
            tsne={"sample_size": 0, "plot": None},
            umap={"sample_size": 0, "plot": None, "error": None},
            anomaly={"count": 0, "percentage": 0, "plot": None},
        )


if __name__ == "__main__":

    app.run(
        debug=True
    )