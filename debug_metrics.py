import pandas as pd


def debug_metrics():
    print("🔍 Debugging metrics calculation...")

    df = pd.read_csv('data/csv_files/synthetic_patient_dataset.csv')

    print(f"Dataset shape: {df.shape}")
    print(f"Treatment groups: {df['Treatment'].unique()}")

    # Check HbA1c values
    print(f"\n HbA1c Statistics:")
    print(f"Overall HbA1c range: {df['HbA1c'].min():.1f}% to {df['HbA1c'].max():.1f}%")
    print(f"Overall HbA1c mean: {df['HbA1c'].mean():.1f}%")

    # Check by treatment group
    ucmsc = df[df['Treatment'] == 'UC_MSC']
    placebo = df[df['Treatment'] == 'Placebo']

    print(f"\n UC-MSC Group:")
    print(f"  Patients: {len(ucmsc)}")
    print(f"  HbA1c range: {ucmsc['HbA1c'].min():.1f}% to {ucmsc['HbA1c'].max():.1f}%")
    print(f"  HbA1c mean: {ucmsc['HbA1c'].mean():.1f}%")
    print(f"  Patients <7%: {(ucmsc['HbA1c'] < 7.0).sum()} ({(ucmsc['HbA1c'] < 7.0).mean() * 100:.1f}%)")

    print(f"\ Placebo Group:")
    print(f"  Patients: {len(placebo)}")
    print(f"  HbA1c range: {placebo['HbA1c'].min():.1f}% to {placebo['HbA1c'].max():.1f}%")
    print(f"  HbA1c mean: {placebo['HbA1c'].mean():.1f}%")
    print(f"  Patients <7%: {(placebo['HbA1c'] < 7.0).sum()} ({(placebo['HbA1c'] < 7.0).mean() * 100:.1f}%)")

    # Check responder rates
    print(f"\nResponder Rates:")
    print(f"  UC-MSC responders: {ucmsc['Responder'].sum()} ({ucmsc['Responder'].mean() * 100:.1f}%)")
    print(f"  Placebo responders: {placebo['Responder'].sum()} ({placebo['Responder'].mean() * 100:.1f}%)")

    # Calculate improvements
    hba1c_improvement = (placebo['HbA1c'].mean() - ucmsc['HbA1c'].mean())
    success_improvement = (ucmsc['HbA1c'] < 7.0).mean() - (placebo['HbA1c'] < 7.0).mean()

    print(f"\nImprovements:")
    print(f"  HbA1c reduction: {hba1c_improvement:.2f}%")
    print(f"  Success rate improvement: {success_improvement * 100:.1f}%")


if __name__ == "__main__":
    debug_metrics()