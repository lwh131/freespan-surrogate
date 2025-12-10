import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import random
from preprocessing import DataScaler
from models import InceptCurvesFiLM
from simulation_utils import beam_on_elastic_foundation_bvp
from sklearn.metrics import f1_score


# Scaler paths must match what was used in training for each model
WINDOW_SEABED_SCALER_PATH = 'seabed_profile_scaler.joblib'
WINDOW_SCALAR_SCALER_PATH = 'feature_scaler.joblib'
WINDOW_TARGETS_SCALER_PATH = 'target_scaler.joblib'
# Paths
TRAINING_BATCH_PATH = 'training_batch'
MASTER_FILE_PATH = 'master_batch_file.parquet'
WINDOW_FILE_PATH = 'master_windows_file.parquet'
WINDOW_MODEL_PATH = 'window_solver.pth'
# Output Directories
WINDOWS_OUTPUT_DIR = 'evaluation_window_plots'
SUMMARY_OUTPUT_DIR = 'evaluation_summary_plots'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
K_FOUNDATION = 1e5

random.seed(123)

def plot_common_profiles(fig, run_data, df_raw):
    '''
    Adds the common, ground-truth traces (seabed and cable profiles) to a Plotly figure.
    
    Args:
        fig (go.Figure): The Plotly figure object to add traces to.
        run_data (pd.Series): The row from the master DataFrame for the specific run.
        df_raw (pd.DataFrame): The raw CSV data for the specific run.
    '''

    # Add the FEA seabed profile - raw
    fig.add_trace(go.Scattergl(
        x=df_raw['seabed_x'], y=df_raw['seabed_z'], mode='lines',
        name='Seabed Profile FEA (Raw)', line=dict(color='red', dash='dash')
    ))
    # Add the FEA cable profile - raw
    fig.add_trace(go.Scattergl(
        x=df_raw['cable_x'], y=df_raw['cable_z'], mode='lines',
        name='Cable Profile FEA (Raw)', line=dict(color='red', dash='dash')
    ))

    # Add the seabed profile - interpolated
    fig.add_trace(go.Scattergl(
        x=run_data['global_x'], y=run_data['seabed_z'], mode='lines',
        name='Seabed Profile (Interpolated)', line=dict(color='grey', dash='dash')
    ))
    # Add the cable profile - interpolated
    fig.add_trace(go.Scattergl(
        x=run_data['global_x'], y=run_data['cable_z'], mode='lines',
        name='Cable Profile (Interpolated)', line=dict(color='dodgerblue', width=3)
    ))


def evaluate_windows(run_id, df_master, df_windows):
    '''
    Loads the window regressor model, evaluates all windows for a single run, generates a plot,
    and returns the true and predicted slope values for aggregation.
    '''
    print(f"\n Evaluating Window Model for Run ID: {run_id} ")
    os.makedirs(WINDOWS_OUTPUT_DIR, exist_ok=True)

    # Load Data and Model
    try:
        df_raw = pd.read_csv(f'{TRAINING_BATCH_PATH}/load_case_{run_id}_results.csv')
    except (FileNotFoundError, IndexError) as e:
        df_raw = None
        # print(f"Error loading files for run {run_id}: {e}")
        return [], [], [], []

    run_data = df_master[df_master['run_id'] == run_id].iloc[0]
    run_windows_data = df_windows[df_windows['run_id'] == run_id]
    if run_windows_data.empty:
        print(f"No windows found for run {run_id} in the window file.")
        return [], [], [], []
    scalar_scaler = DataScaler.load(WINDOW_SCALAR_SCALER_PATH)
    profile_scaler = DataScaler.load(WINDOW_SEABED_SCALER_PATH)
    targets_scaler = DataScaler.load(WINDOW_TARGETS_SCALER_PATH)
    checkpoint = torch.load(WINDOW_MODEL_PATH, map_location=DEVICE)
    model = InceptCurvesFiLM(**checkpoint['model_params']).to(DEVICE) 
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Plotting 
    fig = go.Figure()
    plot_common_profiles(fig, run_data, df_raw)

    # Evaluate and Plot Each Window
    total_mse = 0
    true_slopes_list = []
    predicted_slopes_list = []
    true_zs_list = []
    predicted_zs_list = []

    for _, window_row in run_windows_data.iterrows():
        # Prepare inputs for this specific window
        window_array = np.array(window_row['window_zs']).reshape(1, -1)
        scaled_window = torch.from_numpy(profile_scaler.transform(window_array)).float().to(DEVICE)

        scalar_keys = ['EA', 'EI', 'submerged_weight', 'residual_lay_tension']
        scalar_array = window_row[scalar_keys].to_numpy().reshape(1, -1)
        scaled_scalars = torch.from_numpy(scalar_scaler.transform(scalar_array)).float().to(DEVICE)

        # Get model predictions for slopes
        with torch.no_grad():
            predictions = model(scaled_window, scaled_scalars).cpu().numpy()
        
        predictions = targets_scaler.inverse_transform(predictions)

        predicted_z_a, predicted_z_b, predicted_slope_a, predicted_slope_b = predictions[0]
        # predicted_z_a, predicted_z_b = predictions[0]

        # Aggregate data for summary plots
        true_slope_a = window_row['fea_slope_start']
        true_slope_b = window_row['fea_slope_end']
        true_z_a = window_row['centre_start_z']
        true_z_b = window_row['centre_end_z']

        true_slopes_list.append([true_slope_a, true_slope_b])
        # predicted_slopes_list.append([predicted_slope_a, predicted_slope_b])

        true_zs_list.append([true_z_a, true_z_b])
        predicted_zs_list.append([predicted_z_a, predicted_z_b])

        # Reconstruct the window shape using the predicted slopes and the BVP solver
        window_solution = beam_on_elastic_foundation_bvp(
            x_a=window_row['centre_start_x'],
            y_a=predicted_z_a,
            x_b=window_row['centre_end_x'],
            y_b=predicted_z_b,
            slope_a=predicted_slope_a,
            slope_b=predicted_slope_b,
            T=window_row['residual_lay_tension'],
            EI=window_row['EI'],
            q_weight=window_row['submerged_weight'],
            k_foundation=K_FOUNDATION,
            seabed_x_coords=np.array(window_row['global_x']),
            seabed_z_coords=np.array(window_row['seabed_z'])
        )

        if window_solution:
            # Evaluate the BVP shape and compare to ground truth
            centre_xs_truth = np.array(window_row['centre_xs'])
            reconstructed_zs = window_solution.sol(centre_xs_truth)[0]
            
            # Add reconstructed window to plot
            fig.add_trace(go.Scattergl(
                x=centre_xs_truth, y=reconstructed_zs, mode='lines',
                name=f"Window Centre {window_row['window_id']:.0f} (Predicted)",
                line=dict(color='rgba(255, 0, 255, 0.6)', width=4) # Magenta
            ))
            
            # Calculate MSE for this window
            window_zs_truth = np.interp(centre_xs_truth, run_data['global_x'], run_data['cable_z'])
            mse = np.mean((reconstructed_zs - window_zs_truth)**2)
            total_mse += mse
        else:
            print(f"BVP solver failed for window {window_row['window_id']} in run {run_id}.")

    avg_mse = total_mse / len(run_windows_data) if not run_windows_data.empty else 0
    print(f"Average Window Shape MSE: {avg_mse:.6f}")

    fig.update_layout(
        title_text=f'Window Reconstruction for Run ID: {run_id} | Avg MSE: {avg_mse:.6f}',
        xaxis_title='Position (X) [m]', yaxis_title='Elevation (Z) [m]',
        template='plotly_dark')
    
    output_path = os.path.join(WINDOWS_OUTPUT_DIR, f'run_{run_id}_window_eval.html')
    fig.write_html(output_path)
    fig.show()
    print(f"Saved window evaluation plot to: {output_path}")

    return true_zs_list, predicted_zs_list, true_slopes_list, predicted_slopes_list


def generate_summary_plots(true_slopes, predicted_slopes):
    '''
    Generates and saves aggregate performance plots for the window regression model.
    '''
    print("\nGenerating summary performance plots...")
    os.makedirs(SUMMARY_OUTPUT_DIR, exist_ok=True)

    true_slopes = np.array(true_slopes)
    predicted_slopes = np.array(predicted_slopes)
    residuals = true_slopes - predicted_slopes

    #  1. Actual vs. Predicted Plot (Y=X Plot) 
    fig_avp = go.Figure()
    fig_avp.add_trace(go.Scattergl(
        x=true_slopes[:, 0], y=predicted_slopes[:, 0], mode='markers',
        name='Slope A', marker=dict(color='cyan', opacity=0.7)
    ))
    fig_avp.add_trace(go.Scattergl(
        x=true_slopes[:, 1], y=predicted_slopes[:, 1], mode='markers',
        name='Slope B', marker=dict(color='magenta', opacity=0.7)
    ))
    # Add Y=X line
    min_val = min(true_slopes.min(), predicted_slopes.min())
    max_val = max(true_slopes.max(), predicted_slopes.max())
    fig_avp.add_trace(go.Scattergl(
        x=[min_val, max_val], y=[min_val, max_val], mode='lines',
        name='y = x', line=dict(color='lime', dash='dash')
    ))
    fig_avp.update_layout(
        title='Actual vs. Predicted Slopes',
        xaxis_title='True Value', yaxis_title='Predicted Value',
        template='plotly_dark', legend_title='Slope',
        xaxis=dict(     # Set 1:1 aspect ratio
            scaleanchor='y',  # Link x-axis scale to y-axis
            scaleratio=1,     # Set 1:1 ratio
            range=[min_val, max_val]  # Set the same range for both axes
        ),
        yaxis=dict(
            range=[min_val, max_val]  # Set the same range for both axes
        )
    )
    avp_path = os.path.join(SUMMARY_OUTPUT_DIR, 'actual_vs_predicted.html')
    fig_avp.write_html(avp_path)
    print(f"Saved 'Actual vs. Predicted' plot to: {avp_path}")

    #  2. Residuals vs. Predicted Plot 
    fig_res = go.Figure()
    fig_res.add_trace(go.Scattergl(
        x=predicted_slopes[:, 0], y=residuals[:, 0], mode='markers',
        name='Slope A Residuals', marker=dict(color='cyan', opacity=0.7)
    ))
    fig_res.add_trace(go.Scattergl(
        x=predicted_slopes[:, 1], y=residuals[:, 1], mode='markers',
        name='Slope B Residuals', marker=dict(color='magenta', opacity=0.7)
    ))
    # Add y=0 line
    fig_res.add_shape(type='line', x0=predicted_slopes.min(), y0=0, x1=predicted_slopes.max(), y1=0,
                      line=dict(color='lime', dash='dash'))
    fig_res.update_layout(
        title='Residuals vs. Predicted Slopes',
        xaxis_title='Predicted Value', yaxis_title='Residual (True - Predicted)',
        template='plotly_dark', legend_title='Slope'
    )
    res_path = os.path.join(SUMMARY_OUTPUT_DIR, 'residuals_vs_predicted.html')
    fig_res.write_html(res_path)
    print(f"Saved 'Residuals' plot to: {res_path}")

    #  3. Residual Distribution Histogram 
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=residuals[:, 0], name='Slope A', marker_color='cyan', opacity=0.75))
    fig_hist.add_trace(go.Histogram(x=residuals[:, 1], name='Slope B', marker_color='magenta', opacity=0.75))
    fig_hist.update_layout(
        barmode='overlay',
        title='Distribution of Residuals',
        xaxis_title='Residual Value', yaxis_title='Count',
        template='plotly_dark', legend_title='Slope'
    )
    hist_path = os.path.join(SUMMARY_OUTPUT_DIR, 'residuals_distribution.html')
    fig_hist.write_html(hist_path)
    print(f"Saved 'Residuals Distribution' plot to: {hist_path}")


def evaluate(num_samples, windows):
    '''
    Runs evaluation for all models on a random sample of validation runs.
    '''
    try:
        df_master = pd.read_parquet(MASTER_FILE_PATH)
        df_windows = pd.read_parquet(WINDOW_FILE_PATH)
        print(f"Length of df_windows: {len(df_windows)}")
    except FileNotFoundError as e:
        print(f"Error: Could not find master data file. {e}")
        return

    # validation_runs = df_master[df_master['split'] == 'validation']['run_id'].unique()
    validation_runs = df_windows[df_windows['split'] == 'validation']['run_id'].unique()

    if len(validation_runs) == 0:
        print("No validation runs found.")
        return
        
    random_sample_ids = random.sample(list(validation_runs), k=num_samples)

    # Lists to aggregate results for summary plots
    all_true_slopes = []
    all_predicted_slopes = []
    all_true_zs = []
    all_predicted_zs = []

    for run_id in random_sample_ids:
        if windows:
            true_zs, predicted_zs, true_slopes, predicted_slopes = evaluate_windows(run_id, df_master, df_windows)
            if true_zs: # Check if the list is not empty
                # all_true_slopes.extend(true_slopes)
                # all_predicted_slopes.extend(predicted_slopes)
                all_true_zs.extend(true_zs)
                all_predicted_zs.extend(predicted_zs)

    #  After loop, generate summary plots if data was collected 
    if windows and all_true_slopes:
        generate_summary_plots(all_true_slopes, all_predicted_slopes)


if __name__ == '__main__':
    evaluate(num_samples=5, windows=True)