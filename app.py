from dash import Dash
from dashboard.data_loader import load_all_data, calculate_metrics
from dashboard.layout import create_layout
from dashboard.callbacks import register_callbacks

# Initialize the Dash app with callback exception suppression
app = Dash(__name__, suppress_callback_exceptions=True)

# Load data
print("Starting"
      "g UC-MSC Dashboard...")
print("=" * 50)
data = load_all_data()
df = data['synthetic']
metrics = calculate_metrics(df)

print("=" * 50)
print(" Dashboard initialized successfully!")
print(f" Loaded {len(df)} patients")
print(f" ML models integrated: {len(data.get('model_performance', []))} models")
print(f" Feature importance: {len(data.get('feature_importance', []))} features")
print("=" * 50)

# Create layout
app.layout = create_layout(metrics, df)

# Register callbacks
register_callbacks(app, data)

if __name__ == '__main__':
    print(" Starting server...")
    print(" Dashboard available at: http://127.0.0.1:8050")
    print(" Press Ctrl+C to stop the server")
    print("=" * 50)
    app.run(debug=True, host='127.0.0.1', port=8050)
