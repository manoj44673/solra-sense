from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
CHART_DIR = BASE_DIR / 'static' / 'charts'
CHART_DIR.mkdir(parents=True, exist_ok=True)


def load_processed():
    path = BASE_DIR / 'data' / 'processed' / 'solar_sense_processed.csv'
    if not path.exists():
        from data_loader import build_dataset
        df, _ = build_dataset(save=True)
        return df
    return pd.read_csv(path, parse_dates=['DATE_TIME'])


def make_charts(df=None):
    if df is None:
        df = load_processed()

    # Daily AC generation
    daily = df.set_index('DATE_TIME').resample('D')['AC_POWER'].sum()
    plt.figure(figsize=(11, 5))
    daily.plot()
    plt.title('Daily Solar AC Power Generation')
    plt.xlabel('Date')
    plt.ylabel('AC Power Sum')
    plt.tight_layout()
    plt.savefig(CHART_DIR / 'daily_generation.png', dpi=140)
    plt.close()

    # Irradiation vs AC power sample
    sample = df.sample(min(10000, len(df)), random_state=42)
    plt.figure(figsize=(9, 5))
    plt.scatter(sample['IRRADIATION'], sample['AC_POWER'], s=5, alpha=0.25)
    plt.title('Irradiation vs AC Power')
    plt.xlabel('Irradiation')
    plt.ylabel('AC Power')
    plt.tight_layout()
    plt.savefig(CHART_DIR / 'irradiation_vs_ac.png', dpi=140)
    plt.close()

    # Fault distribution
    counts = df['FAULT_FLAG'].value_counts().sort_index()
    plt.figure(figsize=(6, 5))
    plt.bar(['Normal', 'Potential fault'], [counts.get(0, 0), counts.get(1, 0)])
    plt.title('SolarSense Fault Flag Distribution')
    plt.ylabel('Records')
    plt.tight_layout()
    plt.savefig(CHART_DIR / 'fault_distribution.png', dpi=140)
    plt.close()

    # Average generation by hour
    hourly = df.groupby('HOUR')['AC_POWER'].mean()
    plt.figure(figsize=(9, 5))
    hourly.plot(marker='o')
    plt.title('Average AC Power by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average AC Power')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(CHART_DIR / 'hourly_generation.png', dpi=140)
    plt.close()

    # Temperature vs irradiation
    plt.figure(figsize=(9, 5))
    plt.scatter(sample['AMBIENT_TEMPERATURE'], sample['IRRADIATION'], s=5, alpha=0.25)
    plt.title('Ambient Temperature vs Irradiation')
    plt.xlabel('Ambient Temperature')
    plt.ylabel('Irradiation')
    plt.tight_layout()
    plt.savefig(CHART_DIR / 'temperature_irradiation.png', dpi=140)
    plt.close()

    return [
        'daily_generation.png', 'irradiation_vs_ac.png', 'fault_distribution.png',
        'hourly_generation.png', 'temperature_irradiation.png'
    ]


if __name__ == '__main__':
    print(make_charts())
