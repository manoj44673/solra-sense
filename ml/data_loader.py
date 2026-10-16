from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

GEN_COLS = ['DATE_TIME', 'PLANT_ID', 'SOURCE_KEY', 'DC_POWER', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD']
WEATHER_COLS = ['DATE_TIME', 'PLANT_ID', 'SOURCE_KEY', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']


def load_raw_data():
    generation = pd.concat([
        pd.read_csv(RAW_DIR / 'Plant_1_Generation_Data.csv'),
        pd.read_csv(RAW_DIR / 'Plant_2_Generation_Data.csv')
    ], ignore_index=True)
    weather = pd.concat([
        pd.read_csv(RAW_DIR / 'Plant_1_Weather_Sensor_Data.csv'),
        pd.read_csv(RAW_DIR / 'Plant_2_Weather_Sensor_Data.csv')
    ], ignore_index=True)
    return generation, weather


def build_dataset(save=True):
    generation, weather = load_raw_data()

    # The two source files use slightly different date formats, so mixed parsing is intentional.
    generation['DATE_TIME'] = pd.to_datetime(generation['DATE_TIME'], format='mixed', dayfirst=True, errors='coerce')
    weather['DATE_TIME'] = pd.to_datetime(weather['DATE_TIME'], format='mixed', dayfirst=True, errors='coerce')

    generation = generation.dropna(subset=['DATE_TIME']).copy()
    weather = weather.dropna(subset=['DATE_TIME']).copy()

    # Weather is recorded at plant/timestamp level. Merge on these keys; SOURCE_KEY differs
    # between the generation inverter records and the weather sensor records.
    df = generation.merge(
        weather.drop(columns=['SOURCE_KEY']),
        on=['DATE_TIME', 'PLANT_ID'],
        how='left'
    )

    weather_cols = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']
    df[weather_cols] = df[weather_cols].interpolate(limit_direction='both')
    df[weather_cols] = df[weather_cols].fillna(df[weather_cols].median(numeric_only=True))

    # Time features
    df['HOUR'] = df['DATE_TIME'].dt.hour
    df['MINUTE'] = df['DATE_TIME'].dt.minute
    df['DAY_OF_WEEK'] = df['DATE_TIME'].dt.dayofweek
    df['MONTH'] = df['DATE_TIME'].dt.month
    df['DAY_OF_YEAR'] = df['DATE_TIME'].dt.dayofyear

    # SolarSense fault flag: a derived operational flag, because the supplied Kaggle data
    # contains no ground-truth fault column. Daylight + essentially zero AC output is treated
    # as a potential fault/anomaly.
    df['FAULT_FLAG'] = ((df['IRRADIATION'] > 0.20) & (df['AC_POWER'] <= 1.0)).astype(int)

    # Useful derived feature for dashboard/analysis; not used as the classifier target input.
    dc_nonzero = df['DC_POWER'].where(df['DC_POWER'] != 0)
    df['AC_DC_EFFICIENCY'] = (df['AC_POWER'] / dc_nonzero).fillna(0).clip(0, 2)

    df = df.sort_values(['PLANT_ID', 'SOURCE_KEY', 'DATE_TIME']).reset_index(drop=True)

    if save:
        out = PROCESSED_DIR / 'solar_sense_processed.csv'
        df.to_csv(out, index=False)
        return df, out
    return df, None


if __name__ == '__main__':
    df, out = build_dataset()
    print(f'Processed shape: {df.shape}')
    print(f'Fault count: {int(df.FAULT_FLAG.sum())} ({df.FAULT_FLAG.mean()*100:.2f}%)')
    print(f'Saved: {out}')
