import pandas as pd


def check_columns():
    print("🔍 Checking your data structure...")

    try:
        df = pd.read_csv('data/csv_files/synthetic_patient_dataset.csv')
        print(f"✅ Data loaded: {df.shape}")
        print("\n📋 ALL COLUMNS:")
        for i, col in enumerate(df.columns):
            print(f"  {i + 1:2d}. {col}")

        print(f"\n🔢 First 3 rows:")
        print(df.head(3).to_string())

        # Check for potential treatment columns
        treatment_like = [col for col in df.columns if
                          any(word in col.lower() for word in ['treatment', 'group', 'therapy', 'arm'])]
        print(f"\n🎯 Potential treatment columns: {treatment_like}")

        # Check for potential HbA1c columns
        hba1c_like = [col for col in df.columns if any(word in col.lower() for word in ['hba1c', 'a1c', 'glycated'])]
        print(f"📊 Potential HbA1c columns: {hba1c_like}")

        # Check for potential insulin columns
        insulin_like = [col for col in df.columns if any(word in col.lower() for word in ['insulin', 'dose'])]
        print(f"💉 Potential insulin columns: {insulin_like}")

        if treatment_like:
            print(f"\nTreatment groups found: {df[treatment_like[0]].unique()}")

    except Exception as e:
        print(f" Error: {e}")


if __name__ == "__main__":
    check_columns()