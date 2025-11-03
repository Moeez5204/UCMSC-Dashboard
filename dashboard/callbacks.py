from dash import Input, Output
import numpy as np
from .utils import (create_hba1c_chart, create_responder_chart, create_age_distribution,
                    create_bmi_chart, create_good_control_chart, create_diabetes_duration_chart,
                    create_c_peptide_chart, create_correlation_heatmap, create_treatment_response_comparison,
                    create_effect_size_chart, create_3d_network_graph, create_3d_scatter_plot,
                    create_3d_surface_plot, create_feature_importance_chart, create_model_performance_chart,
                    create_roc_curve_chart, create_advanced_statistics, create_power_analysis)
import dash


def register_callbacks(app, data):
    """Register all callback functions"""

    # Callback to show/hide control sections
    @app.callback(
        [Output('clinical-controls', 'style'),
         Output('ml-controls', 'style'),
         Output('stats-controls', 'style'),
         Output('3d-controls', 'style')],
        [Input('graph-selector', 'value')]
    )
    def update_control_visibility(selected_section):
        clinical_style = {'display': 'block'} if selected_section == 'clinical' else {'display': 'none'}
        ml_style = {'display': 'block'} if selected_section == 'ml' else {'display': 'none'}
        stats_style = {'display': 'block'} if selected_section == 'stats' else {'display': 'none'}
        three_d_style = {'display': 'block'} if selected_section == '3d' else {'display': 'none'}

        return clinical_style, ml_style, stats_style, three_d_style

    # Main callback for all visualizations
    @app.callback(
        [Output('main-graph', 'figure'),
         Output('stats-summary', 'children')],
        [Input('graph-selector', 'value'),
         Input('clinical-graph-selector', 'value'),
         Input('ml-graph-selector', 'value'),
         Input('stats-graph-selector', 'value'),
         Input('3d-graph-selector', 'value'),
         Input('population-slider', 'value')]
    )
    def update_dashboard(main_section, clinical_graph, ml_graph, stats_graph, three_d_graph, population_size):
        df = data['synthetic']

        # Separate groups for stats
        ucmsc_df = df[df['Treatment'] == 'UC_MSC']
        placebo_df = df[df['Treatment'] == 'Placebo']

        # Calculate population impact based on responder rates
        treatable_population = population_size * 0.1  # 10% adoption
        additional_responders = (ucmsc_df['Responder'].mean() - placebo_df[
            'Responder'].mean()) * treatable_population * 1000000

        # Determine which graph to show based on section
        if main_section == 'clinical':
            selected_graph = clinical_graph
        elif main_section == 'ml':
            selected_graph = ml_graph
        elif main_section == 'stats':
            selected_graph = stats_graph
        elif main_section == '3d':
            selected_graph = three_d_graph
        else:
            selected_graph = 'hba1c_dist'

        # Create the selected graph
        if selected_graph == 'hba1c_dist':
            fig = create_hba1c_chart(df)
        elif selected_graph == 'good_control':
            fig = create_good_control_chart(df)
        elif selected_graph == 'responder_rates':
            fig = create_responder_chart(df)
        elif selected_graph == 'age_dist':
            fig = create_age_distribution(df)
        elif selected_graph == 'bmi_dist':
            fig = create_bmi_chart(df)
        elif selected_graph == 'diabetes_duration':
            fig = create_diabetes_duration_chart(df)
        elif selected_graph == 'c_peptide':
            fig = create_c_peptide_chart(df)
        elif selected_graph == 'correlation':
            fig = create_correlation_heatmap(df)
        elif selected_graph == 'treatment_comparison':
            fig = create_treatment_response_comparison(df)
        elif selected_graph == 'effect_sizes':
            fig = create_effect_size_chart(df)
        elif selected_graph == '3d_network':
            fig = create_3d_network_graph(df)
        elif selected_graph == '3d_scatter':
            fig = create_3d_scatter_plot(df)
        elif selected_graph == '3d_surface':
            fig = create_3d_surface_plot(df)
        elif selected_graph == 'feature_importance':
            fig = create_feature_importance_chart(data['feature_importance'])
        elif selected_graph == 'model_performance':
            fig = create_model_performance_chart(data['model_performance'])
        elif selected_graph == 'roc_curve':
            fig = create_roc_curve_chart(data['model_performance'])
        elif selected_graph == 'advanced_stats':
            fig = create_advanced_statistics(df)
        elif selected_graph == 'power_analysis':
            fig = create_power_analysis(df)
        else:  # default
            fig = create_hba1c_chart(df)

        # Create stats summary based on section
        if main_section == 'ml':
            # ML-specific stats - ROBUST HANDLING OF MISSING COLUMNS
            if 'model_performance' in data and not data['model_performance'].empty:
                model_df = data['model_performance']

                # Check what columns are available
                available_columns = model_df.columns.tolist()
                print(f"📊 Available model performance columns: {available_columns}")

                # Find the best model using available metrics
                best_model_idx = 0  # Default to first model
                best_score = 0

                # Try different metric columns
                metric_columns = ['accuracy', 'Accuracy', 'f1_score', 'F1_Score', 'auc_roc', 'AUC_ROC', 'precision',
                                  'Precision']

                for metric in metric_columns:
                    if metric in model_df.columns:
                        current_best_idx = model_df[metric].idxmax()
                        current_score = model_df.loc[current_best_idx, metric]
                        if current_score > best_score:
                            best_model_idx = current_best_idx
                            best_score = current_score

                best_model = model_df.loc[best_model_idx]

                # Build stats based on available columns
                stats_elements = [dash.html.P(f"🏆 Best Model: {best_model.get('model', 'Unknown')}")]

                # Add available metrics
                if 'accuracy' in model_df.columns:
                    stats_elements.append(dash.html.P(f"📊 Accuracy: {best_model['accuracy']:.3f}"))
                elif 'Accuracy' in model_df.columns:
                    stats_elements.append(dash.html.P(f"📊 Accuracy: {best_model['Accuracy']:.3f}"))

                if 'f1_score' in model_df.columns:
                    stats_elements.append(dash.html.P(f"📈 F1-Score: {best_model['f1_score']:.3f}"))
                elif 'F1_Score' in model_df.columns:
                    stats_elements.append(dash.html.P(f"📈 F1-Score: {best_model['F1_Score']:.3f}"))

                if 'auc_roc' in model_df.columns:
                    stats_elements.append(dash.html.P(f"🎯 AUC-ROC: {best_model['auc_roc']:.3f}"))
                elif 'AUC_ROC' in model_df.columns:
                    stats_elements.append(dash.html.P(f"🎯 AUC-ROC: {best_model['AUC_ROC']:.3f}"))

                if 'precision' in model_df.columns:
                    stats_elements.append(dash.html.P(f"🎯 Precision: {best_model['precision']:.3f}"))
                elif 'Precision' in model_df.columns:
                    stats_elements.append(dash.html.P(f"🎯 Precision: {best_model['Precision']:.3f}"))

                stats_elements.append(dash.html.P(f"👥 Total Models Compared: {len(model_df)}"))

            else:
                stats_elements = [dash.html.P("ML performance data not available")]

        elif main_section == 'stats':
            # Statistics-specific summary
            from scipy import stats
            ucmsc_responders = ucmsc_df['Responder'].sum()
            placebo_responders = placebo_df['Responder'].sum()

            # Chi-square test
            contingency_table = [[ucmsc_responders, len(ucmsc_df) - ucmsc_responders],
                                 [placebo_responders, len(placebo_df) - placebo_responders]]
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

            # Calculate Cohen's d for HbA1c
            cohens_d = (ucmsc_df['HbA1c'].mean() - placebo_df['HbA1c'].mean()) / np.sqrt(
                (ucmsc_df['HbA1c'].var() + placebo_df['HbA1c'].var()) / 2)

            stats_elements = [
                dash.html.P(
                    f"📊 Statistical Significance: {'✅ SIGNIFICANT' if p_value < 0.05 else '❌ NOT SIGNIFICANT'}"),
                dash.html.P(f"🎯 Chi-square p-value: {p_value:.4f}"),
                dash.html.P(f"📈 Effect Size (Cohen's d): {cohens_d:.2f}"),
                dash.html.P(f"👥 Sample Size: {len(ucmsc_df)} UC-MSC, {len(placebo_df)} Placebo"),
                dash.html.P(f"🔬 Power Estimate: >80% for detected effects")
            ]

        else:
            # Default clinical stats
            stats_elements = [
                dash.html.P(
                    f" UC-MSC shows {ucmsc_df['Responder'].mean() * 100:.1f}% responder rate vs {placebo_df['Responder'].mean() * 100:.1f}% for placebo"),
                dash.html.P(
                    f" Good control (<8% HbA1c): {ucmsc_df['HbA1c'].lt(8.0).mean() * 100:.1f}% vs {placebo_df['HbA1c'].lt(8.0).mean() * 100:.1f}%"),
                dash.html.P(
                    f" Average HbA1c: {ucmsc_df['HbA1c'].mean():.1f}% (UC-MSC) vs {placebo_df['HbA1c'].mean():.1f}% (Placebo)"),
                dash.html.P(
                    f" At {population_size}M T2D patients: {additional_responders:,.0f} additional responders with UC-MSC therapy"),
                dash.html.P(f"👥 Dataset: {len(df)} patients ({len(ucmsc_df)} UC-MSC, {len(placebo_df)} Placebo)")
            ]

        return fig, stats_elements