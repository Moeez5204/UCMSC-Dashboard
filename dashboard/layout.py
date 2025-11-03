from dash import html, dcc, dash_table


def create_layout(metrics, df):
    """Create the main dashboard layout with ML sections"""

    return html.Div([
        # Header
        html.Div([
            html.H1("🎯 UC-MSC Therapy for Type 2 Diabetes",
                    style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 10}),
            html.P("Interactive Impact Analysis Dashboard",
                   style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': 18}),
        ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': 20}),

        # Key Metrics Row
        html.Div([
            html.Div([
                html.H4("Good Control", style={'color': '#27ae60', 'margin': 0}),
                html.H2(f"{metrics.get('good_control_rate', 0):.1f}%", style={'margin': 0}),
                html.P("Patients <8% HbA1c", style={'color': '#7f8c8d', 'margin': 0})
            ], style=metric_style()),

            html.Div([
                html.H4("Treatment Benefit", style={'color': '#2980b9', 'margin': 0}),
                html.H2(f"+{metrics.get('responder_improvement', 0):.1f}%", style={'margin': 0}),
                html.P("Responders vs Control", style={'color': '#7f8c8d', 'margin': 0})
            ], style=metric_style()),

            html.Div([
                html.H4("Responder Rate", style={'color': '#e74c3c', 'margin': 0}),
                html.H2(f"{metrics.get('treatment_responders', 0):.1f}%", style={'margin': 0}),
                html.P("UC-MSC Responders", style={'color': '#7f8c8d', 'margin': 0})
            ], style=metric_style()),

            html.Div([
                html.H4("HbA1c Levels", style={'color': '#8e44ad', 'margin': 0}),
                html.H2(f"{metrics.get('avg_hba1c_treatment', 0):.1f}%", style={'margin': 0}),
                html.P("UC-MSC Average", style={'color': '#7f8c8d', 'margin': 0})
            ], style=metric_style()),
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': 30}),

        # Controls
        html.Div([
            html.Div([
                html.Label("Select Analysis Section:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='graph-selector',
                    options=[
                        {'label': '📊 Clinical Analysis', 'value': 'clinical'},
                        {'label': '🤖 ML Model Results', 'value': 'ml'},
                        {'label': '📈 Advanced Statistics', 'value': 'stats'},
                        {'label': '🌐 3D Visualizations', 'value': '3d'}
                    ],
                    value='clinical',
                    style={'width': '400px'}
                )
            ], style={'margin': '10px'}),

            # Clinical Visualizations
            html.Div(id='clinical-controls', style={'display': 'block'}, children=[
                html.Div([
                    html.Label("Clinical Visualization:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='clinical-graph-selector',
                        options=[
                            {'label': '📊 HbA1c Distribution', 'value': 'hba1c_dist'},
                            {'label': '🎯 Good Control Rates', 'value': 'good_control'},
                            {'label': '📈 Responder Rates', 'value': 'responder_rates'},
                            {'label': '📋 Treatment Comparison', 'value': 'treatment_comparison'},
                            {'label': '📉 Effect Sizes', 'value': 'effect_sizes'},
                            {'label': '👥 Age Distribution', 'value': 'age_dist'},
                            {'label': '⚖️ BMI Distribution', 'value': 'bmi_dist'},
                            {'label': '🕒 Diabetes Duration', 'value': 'diabetes_duration'},
                            {'label': '🔬 C-Peptide Levels', 'value': 'c_peptide'},
                            {'label': '🔄 Correlation Heatmap', 'value': 'correlation'},
                        ],
                        value='hba1c_dist',
                        style={'width': '400px'}
                    )
                ], style={'margin': '10px'}),
            ]),

            # ML Visualizations
            html.Div(id='ml-controls', style={'display': 'none'}, children=[
                html.Div([
                    html.Label("ML Visualization:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='ml-graph-selector',
                        options=[
                            {'label': '🎯 Feature Importance', 'value': 'feature_importance'},
                            {'label': '📊 Model Performance', 'value': 'model_performance'},
                            {'label': '📈 ROC Curve Comparison', 'value': 'roc_curve'},
                        ],
                        value='feature_importance',
                        style={'width': '400px'}
                    )
                ], style={'margin': '10px'}),
            ]),

            # Statistics Visualizations
            html.Div(id='stats-controls', style={'display': 'none'}, children=[
                html.Div([
                    html.Label("Statistical Analysis:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='stats-graph-selector',
                        options=[
                            {'label': '📊 Advanced Statistics', 'value': 'advanced_stats'},
                            {'label': '📈 Power Analysis', 'value': 'power_analysis'},
                        ],
                        value='advanced_stats',
                        style={'width': '400px'}
                    )
                ], style={'margin': '10px'}),
            ]),

            # 3D Visualizations
            html.Div(id='3d-controls', style={'display': 'none'}, children=[
                html.Div([
                    html.Label("3D Visualization:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='3d-graph-selector',
                        options=[
                            {'label': '🌐 3D Network Graph', 'value': '3d_network'},
                            {'label': '🔵 3D Scatter Plot', 'value': '3d_scatter'},
                            {'label': '📈 3D Response Surface', 'value': '3d_surface'},
                        ],
                        value='3d_network',
                        style={'width': '400px'}
                    )
                ], style={'margin': '10px'}),
            ]),

        ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': 30}),

        # Main Graph Area
        html.Div([
            dcc.Graph(id='main-graph')
        ], style={'marginBottom': 30}),

        # Population Slider (moved underneath the graph)
        html.Div([
            html.Label("Global T2D Population (millions):", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='population-slider',
                min=10,
                max=500,
                step=10,
                value=140,
                marks={i: str(i) for i in [10, 100, 200, 300, 400, 500]},
            )
        ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': 30, 'width': '80%', 'marginLeft': 'auto', 'marginRight': 'auto'}),

        # Statistical Summary
        html.Div([
            html.H3("Statistical Summary", style={'color': '#2c3e50'}),
            html.Div(id='stats-summary')
        ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px'}),

    ], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif'})


def metric_style():
    return {
        'backgroundColor': 'white',
        'padding': '20px',
        'borderRadius': '10px',
        'textAlign': 'center',
        'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
        'flex': 1,
        'margin': '5px'
    }