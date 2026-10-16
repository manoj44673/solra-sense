# SolarSense — Complete Solar Analytics & ML Dashboard

SolarSense is a Flask-based machine-learning web application for solar power
plant analysis. It combines generation and weather-sensor data, performs
complete EDA, compares regression/ensemble models, and provides live
predictions for AC power and a potential daytime underperformance flag.

## Main features

1. **Dashboard** — dataset and model summary.
2. **Data** — processed dataset preview.
3. **EDA** — statistics, missing values, duplicates, unique values, outliers,
   correlations, distributions, time patterns and data-driven findings.
4. **Regression** — Linear Regression, Ridge and Random Forest comparison for
   `AC_POWER`.
5. **Ensemble** — Random Forest, Decision Tree, AdaBoost and optional XGBoost
   comparison for `AC_POWER`.
6. **Predict** — enter plant/inverter/environmental conditions to estimate AC
   generation and potential fault probability.

## Dataset

The project uses four supplied CSV files:

- `data/raw/Plant_1_Generation_Data.csv`
- `data/raw/Plant_1_Weather_Sensor_Data.csv`
- `data/raw/Plant_2_Generation_Data.csv`
- `data/raw/Plant_2_Weather_Sensor_Data.csv`

The combined generation + weather dataset contains **136,476 rows and 11
original columns**. SolarSense preprocessing additionally creates time
features and an operational potential-fault flag.

### Dataset merge

Generation records contain inverter `SOURCE_KEY`; weather records contain a
sensor source key. They represent different entities, so weather `SOURCE_KEY`
is renamed to `SENSOR_KEY` and weather is merged with generation using:

`PLANT_ID + DATE_TIME`

This keeps every generation record while attaching the matching plant-level
weather observation.

## Potential fault definition

The original dataset has **no ground-truth physical fault label**. SolarSense
therefore creates:

`FAULT_FLAG = 1` when `IRRADIATION > 0.20` and `AC_POWER <= 1.0`; otherwise `0`.

This is an operational anomaly/underperformance indicator, **not a confirmed
maintenance diagnosis**.

For the fault classifier, `AC_POWER` is deliberately excluded from model
inputs because it is used to create the label; including it would create target
leakage.

## ML models

### Fault detection
- Target: `FAULT_FLAG`
- Model: Random Forest Classifier
- Inputs: DC power, daily yield, irradiation, temperatures, time features,
  plant and inverter identifiers
- Categorical IDs are encoded with `OrdinalEncoder`.

### Generation prediction
- Target: `AC_POWER`
- Model: Random Forest Regressor for the dashboard prediction form
- Inputs: weather, time, plant and inverter information

### Regression comparison page
Compares:
- Linear Regression
- Ridge Regression
- Random Forest Regressor

Preprocessing:
- `StandardScaler` for numeric environmental features
- `OneHotEncoder` for `PLANT_ID`
- 80/20 train/test split with `random_state=42`

### Ensemble comparison page
Compares:
- Random Forest
- Decision Tree
- AdaBoost
- XGBoost when the optional `xgboost` package is installed

## Important model compatibility fix

The original SolarSense project used serialized `HistGradientBoosting` models.
With a newer scikit-learn installation, the saved model could fail with:

`No module named '_loss'`

The complete version replaces those fragile prediction artifacts with
freshly trained Random Forest prediction models and pins:

`scikit-learn==1.8.0`

The prediction code also automatically rebuilds the prediction models if a
serialized model cannot be loaded.

## Project structure

```text
SolarSense_Complete/
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   └── processed/
│       └── solar_sense_processed.csv
│
├── ml/
│   ├── data_loader.py
│   ├── eda.py
│   └── model.py
│
├── models/
│   ├── load.py
│   ├── eda.py
│   ├── regression.py
│   ├── ensemble.py
│   ├── fault_model.joblib
│   ├── generation_model.joblib
│   └── metrics.json
│
├── trained_models/
│   └── (regression/ensemble pipelines are generated when those pages run)
│
├── static/
│   ├── style.css
│   ├── charts/
│   ├── plots/
│   ├── regression_plots/
│   └── ensemble_plots/
│
└── templates/
    ├── base.html
    ├── index.html
    ├── data.html
    ├── eda.html
    ├── regression.html
    ├── ensemble.html
    └── prediction.html
```

## Installation — Windows

Open the PyCharm Terminal in the project root:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If an older environment already has incompatible scikit-learn installed:

```powershell
python -m pip uninstall scikit-learn -y
pip install scikit-learn==1.8.0
```

Verify:

```powershell
python -c "import sklearn; print(sklearn.__version__)"
```

It should print:

```text
1.8.0
```

## Run

```powershell
python app.py
```

Open:

`http://127.0.0.1:5000`

## Useful routes

- `/` — Dashboard
- `/data` — Data preview
- `/eda` — Full EDA
- `/regression` — Regression comparison
- `/ensemble` — Ensemble comparison
- `/predict` — SolarSense generation + potential fault prediction

## Model results included with this version

The prediction models were freshly trained using the same deterministic
split/sample settings as the original project.

- Fault accuracy: **1.00**
- Fault precision: **1.00**
- Fault recall: **1.00**
- Fault F1: **1.00**
- Generation MAE: **39.7617**
- Generation RMSE: **122.4507**
- Generation R²: **0.8963**

The perfect fault score must be interpreted carefully: the target is a
derived rule rather than a ground-truth maintenance label.

## Viva explanation

> SolarSense is a machine-learning-based solar monitoring and prediction
> system. It combines solar generation and weather data, cleans and analyzes
> the data, performs EDA, trains regression and ensemble models, and provides
> a Flask web interface. The system predicts AC power as a regression task and
> identifies potential daytime underperformance using a derived binary fault
> flag. The results are presented through Dashboard, Data, EDA, Regression,
> Ensemble and Prediction pages.

## Future scope

- Use real maintenance/fault labels from solar plants.
- Add real-time IoT sensor streams.
- Add time-series forecasting.
- Add plant-level anomaly alerts.
- Deploy the application to a cloud platform.
- Add model monitoring and scheduled retraining.
