import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import stats


# Basic Chart Functions
def create_hba1c_chart(df):
    """Create HbA1c distribution chart for your data"""
    fig = px.box(df, x='Treatment', y='HbA1c',
                 title='HbA1c Distribution by Treatment Group',
                 color='Treatment')
    fig.add_hline(y=7.0, line_dash="dash", line_color="red",
                  annotation_text="Target: <7.0%")
    fig.add_hline(y=8.0, line_dash="dash", line_color="orange",
                  annotation_text="Good: <8.0%")
    return fig


def create_good_control_chart(df):
    """Create good control rate chart (<8% HbA1c)"""
    control_data = df.groupby('Treatment')['HbA1c'].apply(
        lambda x: (x < 8.0).mean() * 100
    ).reset_index()
    control_data.columns = ['Treatment', 'Good_Control_Rate']

    fig = px.bar(control_data, x='Treatment', y='Good_Control_Rate',
                 title='Good Control Rate (<8% HbA1c) by Treatment Group',
                 color='Treatment', text_auto=True)
    fig.update_layout(yaxis_title="Good Control Rate (%)")
    return fig


def create_responder_chart(df):
    """Create responder rate chart"""
    responder_data = df.groupby('Treatment')['Responder'].mean() * 100
    responder_data = responder_data.reset_index()
    responder_data.columns = ['Treatment', 'Responder_Rate']

    fig = px.bar(responder_data, x='Treatment', y='Responder_Rate',
                 title='Treatment Responder Rates',
                 color='Treatment', text_auto=True)
    fig.update_layout(yaxis_title="Responder Rate (%)")
    return fig


def create_age_distribution(df):
    """Create age distribution chart"""
    fig = px.histogram(df, x='Age', color='Treatment',
                       title='Age Distribution by Treatment Group',
                       barmode='overlay', opacity=0.7)
    fig.update_layout(xaxis_title="Age (years)", yaxis_title="Number of Patients")
    return fig


def create_bmi_chart(df):
    """Create BMI distribution chart"""
    fig = px.box(df, x='Treatment', y='BMI',
                 title='BMI Distribution by Treatment Group',
                 color='Treatment')
    fig.update_layout(yaxis_title="BMI")
    return fig


def create_diabetes_duration_chart(df):
    """Create diabetes duration distribution chart"""
    fig = px.box(df, x='Treatment', y='Diabetes_Duration',
                 title='Diabetes Duration by Treatment Group',
                 color='Treatment')
    fig.update_layout(yaxis_title="Diabetes Duration (years)")
    return fig


def create_c_peptide_chart(df):
    """Create C-Peptide distribution chart"""
    fig = px.box(df, x='Treatment', y='C_Peptide',
                 title='C-Peptide Levels by Treatment Group',
                 color='Treatment')
    fig.update_layout(yaxis_title="C-Peptide Level")
    return fig


def create_correlation_heatmap(df):
    """Create correlation heatmap of numeric variables"""
    numeric_df = df.select_dtypes(include=['number'])

    # Remove Patient_ID if it exists
    if 'Patient_ID' in numeric_df.columns:
        numeric_df = numeric_df.drop('Patient_ID', axis=1)

    if len(numeric_df.columns) > 1:
        correlation_matrix = numeric_df.corr()
        fig = px.imshow(correlation_matrix,
                        title='Correlation Heatmap of Clinical Variables',
                        aspect='auto',
                        color_continuous_scale='RdBu_r')
        return fig
    else:
        fig = go.Figure()
        fig.add_annotation(text="Not enough numeric variables for correlation matrix",
                           x=0.5, y=0.5, showarrow=False)
        return fig


def create_treatment_response_comparison(df):
    """Create side-by-side comparison of key metrics"""
    # Calculate metrics for each treatment group
    metrics_data = []

    for treatment in df['Treatment'].unique():
        treatment_df = df[df['Treatment'] == treatment]

        metrics_data.append({
            'Treatment': treatment,
            'Avg_HbA1c': treatment_df['HbA1c'].mean(),
            'Responder_Rate': treatment_df['Responder'].mean() * 100,
            'Good_Control_Rate': (treatment_df['HbA1c'] < 8.0).mean() * 100,
            'Avg_BMI': treatment_df['BMI'].mean(),
            'Avg_Age': treatment_df['Age'].mean(),
            'Avg_Diabetes_Duration': treatment_df['Diabetes_Duration'].mean()
        })

    metrics_df = pd.DataFrame(metrics_data)

    # Create subplots
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Average HbA1c', 'Responder Rate', 'Good Control Rate', 'Average BMI'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )

    # HbA1c
    fig.add_trace(
        go.Bar(x=metrics_df['Treatment'], y=metrics_df['Avg_HbA1c'],
               name='Avg HbA1c', marker_color='lightcoral'),
        row=1, col=1
    )

    # Responder Rate
    fig.add_trace(
        go.Bar(x=metrics_df['Treatment'], y=metrics_df['Responder_Rate'],
               name='Responder Rate', marker_color='lightblue'),
        row=1, col=2
    )

    # Good Control Rate
    fig.add_trace(
        go.Bar(x=metrics_df['Treatment'], y=metrics_df['Good_Control_Rate'],
               name='Good Control', marker_color='lightgreen'),
        row=2, col=1
    )

    # BMI
    fig.add_trace(
        go.Bar(x=metrics_df['Treatment'], y=metrics_df['Avg_BMI'],
               name='Avg BMI', marker_color='lightsalmon'),
        row=2, col=2
    )

    fig.update_layout(height=600, title_text="Treatment Group Comparison")
    fig.update_yaxes(title_text="HbA1c (%)", row=1, col=1)
    fig.update_yaxes(title_text="Rate (%)", row=1, col=2)
    fig.update_yaxes(title_text="Rate (%)", row=2, col=1)
    fig.update_yaxes(title_text="BMI", row=2, col=2)

    return fig


def create_effect_size_chart(df):
    """Create chart showing effect sizes for key outcomes"""
    effect_data = []

    # Calculate effect sizes for key metrics
    metrics_to_compare = [
        ('HbA1c', 'lower', 'HbA1c Reduction'),
        ('Responder', 'higher', 'Responder Rate'),
        ('BMI', 'neutral', 'BMI'),
        ('Age', 'neutral', 'Age'),
        ('Diabetes_Duration', 'neutral', 'Diabetes Duration')
    ]

    for metric, direction, label in metrics_to_compare:
        if metric in df.columns:
            ucmsc_mean = df[df['Treatment'] == 'UC_MSC'][metric].mean()
            placebo_mean = df[df['Treatment'] == 'Placebo'][metric].mean()

            if metric == 'Responder':
                # For binary outcomes, use risk difference
                effect_size = ucmsc_mean - placebo_mean
            else:
                # For continuous outcomes, use mean difference
                effect_size = ucmsc_mean - placebo_mean
                if direction == 'lower':
                    effect_size = -effect_size  # Make negative values "good"

            effect_data.append({
                'Metric': label,
                'Effect_Size': effect_size,
                'Direction': direction
            })

    effect_df = pd.DataFrame(effect_data)

    # Color based on direction and effect
    colors = []
    for _, row in effect_df.iterrows():
        if row['Direction'] == 'higher' and row['Effect_Size'] > 0:
            colors.append('#27ae60')  # Green for positive effects
        elif row['Direction'] == 'lower' and row['Effect_Size'] > 0:
            colors.append('#27ae60')  # Green for positive effects
        elif row['Effect_Size'] < 0:
            colors.append('#e74c3c')  # Red for negative effects
        else:
            colors.append('#95a5a6')  # Gray for neutral/no effect

    fig = px.bar(effect_df, x='Effect_Size', y='Metric', orientation='h',
                 title='Treatment Effect Sizes (UC-MSC vs Placebo)',
                 color_discrete_sequence=colors)

    fig.update_layout(xaxis_title="Effect Size", yaxis_title="")
    fig.add_vline(x=0, line_dash="dash", line_color="black")

    return fig


# 3D Visualization Functions
def create_3d_network_graph():
    """Create interactive 3D network graph showing relationships"""
    import networkx as nx

    # Create a graph
    G = nx.Graph()

    # Add nodes for key biomarkers and outcomes with better positioning
    nodes = [
        ('HbA1c', 'biomarker', 25, 0, 0, 0),
        ('BMI', 'biomarker', 20, 2, 0, 0),
        ('Age', 'demographic', 15, 0, 2, 0),
        ('Diabetes_Duration', 'demographic', 15, 0, 0, 2),
        ('C_Peptide', 'biomarker', 20, -2, 0, 0),
        ('Responder', 'outcome', 30, 0, -2, 0),
        ('UC-MSC', 'treatment', 35, 2, 2, 2),
        ('Placebo', 'treatment', 25, -2, -2, -2)
    ]

    for node, node_type, size, x, y, z in nodes:
        G.add_node(node, type=node_type, size=size, pos=(x, y, z))

    # Add guaranteed edges for visibility
    G.add_edge('HbA1c', 'BMI', weight=8, type='correlation')
    G.add_edge('HbA1c', 'Responder', weight=10, type='correlation')
    G.add_edge('UC-MSC', 'Responder', weight=12, type='treatment')
    G.add_edge('UC-MSC', 'HbA1c', weight=8, type='treatment')
    G.add_edge('Placebo', 'HbA1c', weight=4, type='control')

    # Use manual positions for better layout
    pos = {}
    for node, node_type, size, x, y, z in nodes:
        pos[node] = [x, y, z]

    # Extract node positions
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_z = [pos[node][2] for node in G.nodes()]

    # Node sizes and colors
    node_sizes = [G.nodes[node]['size'] for node in G.nodes()]
    node_colors = []
    node_text = []
    for node in G.nodes():
        node_type = G.nodes[node]['type']
        if node_type == 'treatment':
            node_colors.append('#FF6B6B' if 'UC-MSC' in node else '#4ECDC4')
        elif node_type == 'outcome':
            node_colors.append('#FFE66D')
        elif node_type == 'biomarker':
            node_colors.append('#45B7D1')
        else:
            node_colors.append('#96CEB4')

        # Add hover text
        node_text.append(f"{node}<br>Type: {node_type}")

    # Create edge traces - FIXED: Create separate traces for each edge
    edge_traces = []

    for edge in G.edges():
        node1, node2 = edge
        x0, y0, z0 = pos[node1]
        x1, y1, z1 = pos[node2]

        # Edge color based on type
        edge_data = G.edges[edge]
        if edge_data.get('type') == 'treatment':
            color = 'rgba(76, 175, 80, 0.8)'  # Green
            width = edge_data['weight']
        elif edge_data.get('type') == 'correlation':
            color = 'rgba(70, 130, 180, 0.8)'  # Blue
            width = edge_data['weight']
        else:
            color = 'rgba(128, 128, 128, 0.5)'  # Gray
            width = edge_data['weight']

        # Create a separate trace for each edge
        edge_trace = go.Scatter3d(
            x=[x0, x1, None],
            y=[y0, y1, None],
            z=[z0, z1, None],
            line=dict(
                width=width,  # Single number, not a list
                color=color
            ),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        )
        edge_traces.append(edge_trace)

    # Create node trace
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='darkgray')
        ),
        textposition="middle center",
        name='Variables'
    )

    # Combine all traces
    all_traces = edge_traces + [node_trace]
    fig = go.Figure(data=all_traces)

    fig.update_layout(
        title='3D Network: Biomarker Relationships and Treatment Effects',
        scene=dict(
            xaxis=dict(
                showbackground=False,
                showticklabels=False,
                title='',
                gridcolor='lightgray',
                gridwidth=1
            ),
            yaxis=dict(
                showbackground=False,
                showticklabels=False,
                title='',
                gridcolor='lightgray',
                gridwidth=1
            ),
            zaxis=dict(
                showbackground=False,
                showticklabels=False,
                title='',
                gridcolor='lightgray',
                gridwidth=1
            ),
            bgcolor='white'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=700,
        showlegend=True
    )

    return fig


def create_3d_scatter_plot(df):
    """Create 3D scatter plot of key variables"""
    fig = px.scatter_3d(df,
                        x='HbA1c',
                        y='BMI',
                        z='Age',
                        color='Treatment',
                        size='Diabetes_Duration',
                        hover_name='Patient_ID',
                        title='3D Scatter: HbA1c vs BMI vs Age by Treatment',
                        opacity=0.7)

    fig.update_layout(
        scene=dict(
            xaxis_title='HbA1c (%)',
            yaxis_title='BMI',
            zaxis_title='Age (years)'
        )
    )

    return fig


def create_3d_surface_plot(df):
    """Create 3D surface plot showing response probability"""
    # Create a grid for surface plot

    hba1c_range = np.linspace(df['HbA1c'].min(), df['HbA1c'].max(), 20)
    bmi_range = np.linspace(df['BMI'].min(), df['BMI'].max(), 20)

    # Simple model: probability of being a responder based on HbA1c and BMI
    # This is a simplified demonstration
    H, B = np.meshgrid(hba1c_range, bmi_range)

    # Example response surface (in reality, you'd use your ML model)
    Z = 1 / (1 + np.exp(-(0.5 * (10 - H) + 0.1 * (25 - B))))

    fig = go.Figure(data=[go.Surface(z=Z, x=hba1c_range, y=bmi_range)])

    fig.update_layout(
        title='3D Response Surface: Estimated Probability of Response',
        scene=dict(
            xaxis_title='HbA1c (%)',
            yaxis_title='BMI',
            zaxis_title='Response Probability',
            zaxis=dict(range=[0, 1])
        ),
        height=600
    )

    return fig


# ML Visualization Functions - UPDATED FOR YOUR SPECIFIC DATA
def create_feature_importance_chart(feature_importance_df):
    """Create feature importance visualization - UPDATED FOR YOUR DATA"""
    if feature_importance_df is None or feature_importance_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Feature importance data not available", x=0.5, y=0.5, showarrow=False)
        return fig

    # Your data has: ['feature', 'importance']
    if 'feature' in feature_importance_df.columns and 'importance' in feature_importance_df.columns:
        # Sort by importance
        df_sorted = feature_importance_df.sort_values('importance', ascending=True)

        fig = px.bar(df_sorted,
                     x='importance',
                     y='feature',
                     title='Feature Importance for Treatment Response Prediction',
                     orientation='h',
                     color='importance',
                     color_continuous_scale='Viridis')

        fig.update_layout(yaxis_title="Features", xaxis_title="Importance Score")
        return fig
    else:
        # If columns don't match, show as table
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(feature_importance_df.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[feature_importance_df[col] for col in feature_importance_df.columns],
                       fill_color='lavender',
                       align='left'))
        ])
        fig.update_layout(title='Feature Importance Data')
        return fig


def create_model_performance_chart(model_performance_df):
    """Create model performance comparison chart - UPDATED FOR YOUR DATA"""
    if model_performance_df is None or model_performance_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Model performance data not available", x=0.5, y=0.5, showarrow=False)
        return fig

    # Your data has: ['Accuracy', 'AUC_ROC', 'Training_Samples', 'Test_Samples']
    # Since you only have one model, create a metrics display

    if len(model_performance_df) == 1:
        # Single model - create a gauge-style visualization
        model_data = model_performance_df.iloc[0]

        # Create gauge charts for Accuracy and AUC-ROC
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "indicator"}, {"type": "indicator"}]],
            subplot_titles=('Accuracy', 'AUC-ROC')
        )

        # Accuracy gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=model_data['Accuracy'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Accuracy"},
            delta={'reference': 0.5, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 0.5], 'color': 'lightcoral'},
                    {'range': [0.5, 0.8], 'color': 'lightyellow'},
                    {'range': [0.8, 1], 'color': 'lightgreen'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.9}}
        ), row=1, col=1)

        # AUC-ROC gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=model_data['AUC_ROC'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "AUC-ROC"},
            delta={'reference': 0.5, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 0.5], 'color': 'lightcoral'},
                    {'range': [0.5, 0.7], 'color': 'lightyellow'},
                    {'range': [0.7, 1], 'color': 'lightgreen'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.8}}
        ), row=1, col=2)

        fig.update_layout(
            title='Model Performance Metrics',
            height=400,
            margin=dict(l=50, r=50, t=80, b=50)
        )

        # Add sample info as annotation
        fig.add_annotation(
            text=f"Training Samples: {model_data['Training_Samples']}<br>Test Samples: {model_data['Test_Samples']}",
            x=0.5, y=0.1, showarrow=False,
            xref="paper", yref="paper",
            font=dict(size=12),
            align="center",
            bgcolor="lightgray"
        )

        return fig
    else:
        # If multiple models, show comparison
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(model_performance_df.columns),
                        fill_color='lightblue',
                        align='left'),
            cells=dict(values=[model_performance_df[col] for col in model_performance_df.columns],
                       fill_color='lavender',
                       align='left'))
        ])
        fig.update_layout(title='Model Performance Data')
        return fig


def create_roc_curve_chart(model_performance_df):
    """Create ROC curve comparison chart - UPDATED FOR YOUR DATA"""
    if model_performance_df is None or model_performance_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Model performance data not available", x=0.5, y=0.5, showarrow=False)
        return fig

    # Your data has: ['Accuracy', 'AUC_ROC', 'Training_Samples', 'Test_Samples']
    # Since we don't have actual ROC curve data, create an informative display

    if len(model_performance_df) == 1:
        model_data = model_performance_df.iloc[0]

        # Create a simulated ROC curve based on AUC-ROC value
        auc_value = model_data['AUC_ROC']

        # Generate points for ROC curve
        fpr = np.linspace(0, 1, 100)

        # Simple approximation of ROC curve shape based on AUC
        if auc_value > 0.5:
            tpr = 1 - (1 - fpr) ** (1 / (2 * auc_value))
        else:
            tpr = fpr ** (1 / (2 * (1 - auc_value)))

        # Ensure curve goes through (0,0) and (1,1)
        tpr[0] = 0
        tpr[-1] = 1

        fig = go.Figure()

        # Add ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'Model (AUC = {auc_value:.3f})',
            line=dict(width=3, color='blue')
        ))

        # Add random classifier line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='red', width=2)
        ))

        fig.update_layout(
            title='ROC Curve (Simulated based on AUC-ROC)',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            showlegend=True,
            width=600,
            height=500
        )

        # Add performance metrics as annotation
        fig.add_annotation(
            text=f"Accuracy: {model_data['Accuracy']:.3f}<br>AUC-ROC: {auc_value:.3f}",
            x=0.95, y=0.05, showarrow=False,
            xref="paper", yref="paper",
            font=dict(size=12),
            align="right",
            bgcolor="lightyellow",
            bordercolor="black",
            borderwidth=1
        )

        return fig
    else:
        # If multiple models, show comparison table
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(model_performance_df.columns),
                        fill_color='lightgreen',
                        align='left'),
            cells=dict(values=[model_performance_df[col] for col in model_performance_df.columns],
                       fill_color='mintcream',
                       align='left'))
        ])
        fig.update_layout(title='Model Performance Comparison')
        return fig


# Advanced Statistics Functions
def create_advanced_statistics(df):
    """Create advanced statistical analysis"""
    ucmsc_df = df[df['Treatment'] == 'UC_MSC']
    placebo_df = df[df['Treatment'] == 'Placebo']

    # T-test for HbA1c
    hba1c_t_stat, hba1c_p_value = stats.ttest_ind(ucmsc_df['HbA1c'], placebo_df['HbA1c'])

    # Chi-square test for responder rates
    contingency_table = pd.crosstab(df['Treatment'], df['Responder'])
    chi2_stat, chi2_p_value, dof, expected = stats.chi2_contingency(contingency_table)

    # Effect sizes
    hba1c_cohens_d = (ucmsc_df['HbA1c'].mean() - placebo_df['HbA1c'].mean()) / np.sqrt(
        (ucmsc_df['HbA1c'].var() + placebo_df['HbA1c'].var()) / 2)

    # Create summary table
    stats_data = {
        'Test': ['HbA1c T-test', 'Responder Chi-square', 'HbA1c Effect Size'],
        'Statistic': [f"{hba1c_t_stat:.3f}", f"{chi2_stat:.3f}", f"{hba1c_cohens_d:.3f}"],
        'P-value': [f"{hba1c_p_value:.4f}", f"{chi2_p_value:.4f}", 'N/A'],
        'Significance': [hba1c_p_value < 0.05, chi2_p_value < 0.05, abs(hba1c_cohens_d) > 0.5]
    }

    stats_df = pd.DataFrame(stats_data)

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(stats_df.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[stats_df[col] for col in stats_df.columns],
                   fill_color='lavender',
                   align='left'))
    ])

    fig.update_layout(title='Advanced Statistical Analysis')
    return fig


def create_power_analysis(df):
    """Create statistical power analysis"""
    ucmsc_df = df[df['Treatment'] == 'UC_MSC']
    placebo_df = df[df['Treatment'] == 'Placebo']

    # Sample size
    n_ucmsc = len(ucmsc_df)
    n_placebo = len(placebo_df)

    # Effect size for responder rate
    p1 = ucmsc_df['Responder'].mean()
    p2 = placebo_df['Responder'].mean()
    effect_size = abs(p1 - p2)

    # Power analysis (simplified)
    total_n = n_ucmsc + n_placebo
    power_estimate = min(0.95, effect_size * total_n / 100)  # Simplified calculation

    power_data = {
        'Metric': ['Sample Size (UC-MSC)', 'Sample Size (Placebo)', 'Total Sample Size',
                   'Responder Rate Difference', 'Estimated Power'],
        'Value': [n_ucmsc, n_placebo, total_n, f"{effect_size * 100:.1f}%", f"{power_estimate * 100:.1f}%"]
    }

    power_df = pd.DataFrame(power_data)

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(power_df.columns),
                    fill_color='lightgreen',
                    align='left'),
        cells=dict(values=[power_df[col] for col in power_df.columns],
                   fill_color='mintcream',
                   align='left'))
    ])

    fig.update_layout(title='Statistical Power Analysis')
    return fig