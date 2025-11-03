import pandas as pd
import numpy as np


def load_all_data():
    """Load and prepare all data files including ML results"""
    data = {}

    try:
        # Load main dataset
        data['synthetic'] = pd.read_csv('data/csv_files/synthetic_patient_dataset.csv')
        print("✅ synthetic_patient_dataset.csv loaded")

        # Load ML results
        try:
            data['feature_importance'] = pd.read_csv('data/csv_files/feature_importance.csv')
            print("✅ feature_importance.csv loaded")
        except:
            print("⚠️  feature_importance.csv not found - creating sample data")
            data['feature_importance'] = create_sample_feature_importance()

        try:
            data['model_performance'] = pd.read_csv('data/csv_files/model_performance.csv')
            print("✅ model_performance.csv loaded")
        except:
            print("⚠️  model_performance.csv not found - creating sample data")
            data['model_performance'] = create_sample_model_performance()

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        data['synthetic'] = create_sample_data()
        data['feature_importance'] = create_sample_feature_importance()
        data['model_performance'] = create_sample_model_performance()

    return data


def create_sample_data():
    """Create sample data if files not found"""
    print("📝 Creating sample data for demo...")
    return pd.DataFrame({
        'Patient_ID': range(200),
        'Treatment': ['UC_MSC'] * 100 + ['Placebo'] * 100,
        'Age': np.random.normal(55, 10, 200),
        'BMI': np.random.normal(28, 5, 200),
        'Diabetes_Duration': np.random.normal(8, 4, 200),
        'HbA1c': np.concatenate([
            np.random.normal(7.0, 1.2, 100),  # Treatment
            np.random.normal(8.2, 1.4, 100)  # Control
        ]),
        'C_Peptide': np.random.normal(2.1, 0.5, 200),
        'Responder': np.concatenate([
            np.random.choice([0, 1], 100, p=[0.6, 0.4]),  # Treatment
            np.random.choice([0, 1], 100, p=[0.9, 0.1])  # Control
        ])
    })


def create_sample_feature_importance():
    """Create sample feature importance data"""
    return pd.DataFrame({
        'feature': ['HbA1c', 'BMI', 'Age', 'Diabetes_Duration', 'C_Peptide', 'Treatment'],
        'importance': [0.35, 0.25, 0.15, 0.12, 0.08, 0.05],
        'model': ['Random Forest'] * 6
    })


def create_sample_model_performance():
    """Create sample model performance data"""
    return pd.DataFrame({
        'model': ['Random Forest', 'Neural Network', 'Logistic Regression'],
        'accuracy': [0.85, 0.82, 0.78],
        'precision': [0.83, 0.81, 0.76],
        'recall': [0.84, 0.80, 0.77],
        'f1_score': [0.835, 0.805, 0.765],
        'auc_roc': [0.89, 0.86, 0.82]
    })


def calculate_metrics(df):
    """Calculate key performance metrics for your actual data"""
    metrics = {}

    print("\n🔍 Calculating metrics from your data...")

    # Separate treatment groups
    ucmsc_df = df[df['Treatment'] == 'UC_MSC']
    placebo_df = df[df['Treatment'] == 'Placebo']

    print(f"   UC-MSC patients: {len(ucmsc_df)}, Placebo patients: {len(placebo_df)}")

    # Since no patients have HbA1c < 7%, use more realistic targets
    # Target 1: HbA1c < 8% (good control)
    # Target 2: HbA1c reduction > 1% (clinically significant)

    # Calculate HbA1c < 8% success rates
    ucmsc_good_control = (ucmsc_df['HbA1c'] < 8.0).mean() * 100
    placebo_good_control = (placebo_df['HbA1c'] < 8.0).mean() * 100

    metrics['good_control_rate'] = (df['HbA1c'] < 8.0).mean() * 100
    metrics['treatment_good_control'] = ucmsc_good_control
    metrics['control_good_control'] = placebo_good_control
    metrics['control_improvement'] = ucmsc_good_control - placebo_good_control

    # Calculate responder rates (from your Responder column)
    ucmsc_responder_rate = ucmsc_df['Responder'].mean() * 100
    placebo_responder_rate = placebo_df['Responder'].mean() * 100

    metrics['treatment_responders'] = ucmsc_responder_rate
    metrics['control_responders'] = placebo_responder_rate
    metrics['responder_improvement'] = ucmsc_responder_rate - placebo_responder_rate

    # Calculate average HbA1c and reduction
    ucmsc_avg_hba1c = ucmsc_df['HbA1c'].mean()
    placebo_avg_hba1c = placebo_df['HbA1c'].mean()

    metrics['avg_hba1c_treatment'] = ucmsc_avg_hba1c
    metrics['avg_hba1c_control'] = placebo_avg_hba1c
    metrics['hba1c_reduction'] = placebo_avg_hba1c - ucmsc_avg_hba1c

    # Calculate patients achieving >1% HbA1c reduction from baseline
    # Since we don't have baseline, we'll estimate based on your data distribution
    # Assuming average baseline was around 9.5% based on typical T2D patients

    estimated_baseline = 9.5
    ucmsc_reduction = estimated_baseline - ucmsc_avg_hba1c
    placebo_reduction = estimated_baseline - placebo_avg_hba1c

    metrics['estimated_reduction_treatment'] = ucmsc_reduction
    metrics['estimated_reduction_control'] = placebo_reduction
    metrics['additional_reduction'] = ucmsc_reduction - placebo_reduction

    metrics['total_patients'] = len(df)

    print(f"   Good control (<8%): UC-MSC {ucmsc_good_control:.1f}%, Placebo {placebo_good_control:.1f}%")
    print(f"   Responder rates: UC-MSC {ucmsc_responder_rate:.1f}%, Placebo {placebo_responder_rate:.1f}%")
    print(f"   Avg HbA1c: UC-MSC {ucmsc_avg_hba1c:.1f}%, Placebo {placebo_avg_hba1c:.1f}%")

    return metrics