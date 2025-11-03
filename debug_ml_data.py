import pandas as pd
import os


def debug_ml_files():
    print(" Debugging ML data files")

    # Check if files exist
    ml_files = {
        'feature_importance': 'data/csv_files/feature_importance.csv',
        'model_performance': 'data/csv_files/model_performance.csv'
    }

    for file_name, file_path in ml_files.items():
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"✅ {file_name}.csv found:")
                print(f"   Shape: {df.shape}")
                print(f"   Columns: {list(df.columns)}")
                if not df.empty:
                    print(f"   First few rows:")
                    print(df.head(3).to_string())
                else:
                    print("File is empty")
            except Exception as e:
                print(f"Error reading {file_name}.csv: {e}")
        else:
            print(f"{file_name}.csv not found at: {file_path}")

        print()


if __name__ == "__main__":
    debug_ml_files()