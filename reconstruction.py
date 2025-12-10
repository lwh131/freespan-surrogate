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
from simulation_utils import generate_seabed
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEABED_SCALER_PATH = 'seabed_profile_scaler.joblib'
FEATURE_SCALER_PATH = 'feature_scaler.joblib'
TARGETS_SCALER_PATH = 'target_scaler.joblib'
MODEL_PATH = 'window_solver.pth'
OUTPUT_DIR = 'reconstruction_plots'

DETERMINISTIC = True

if DETERMINISTIC:
    SEED = 1234
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    if DEVICE == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_error_histogram(errors, run_data, output_dir):
    """
    Creates and saves a Matplotlib histogram of the prediction errors.
    """
    run_id = run_data['run_id']
    seabed_id = run_data['seabed_id']
    synthetic_type = run_data.get('synthetic', 'Generated')

    mean_error = np.mean(errors)
    std_error = np.std(errors)

    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.75, color='cornflowerblue', edgecolor='black')
    
    plt.axvline(mean_error, color='r', linestyle='dashed', linewidth=2, label=f'Mean Error: {mean_error:.4f} m')
    plt.axvline(0, color='k', linestyle='solid', linewidth=1, label='Zero Error')

    plt.title(f'Prediction Error Distribution for Run ID: {run_id}\n(Seabed: {seabed_id}, Type: {synthetic_type})')
    plt.xlabel('Prediction Error (Predicted Z - True Z) [m]')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    stats_text = f'Mean: {mean_error:.4f} m\nStd Dev: {std_error:.4f} m\nN points: {len(errors)}'
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    output_path = os.path.join(output_dir, f'run_{run_id}_error_histogram.png')
    plt.savefig(output_path, dpi=300)
    print(f"    Saved error histogram to: {output_path}")
    # plt.show()
    plt.close()


def plot_parity(true_z, predicted_z, run_data, output_dir):
    """
    Creates and saves a Matplotlib parity plot (predicted vs. true values).
    """
    run_id = run_data['run_id']
    seabed_id = run_data['seabed_id']
    synthetic_type = run_data.get('synthetic', 'Generated')

    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    ax.scatter(true_z, predicted_z, alpha=0.5, s=10, label='Data Points')

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction (y=x)')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('True Elevation (Z) [m]')
    ax.set_ylabel('Predicted Elevation (Z) [m]')
    ax.set_title(f'Parity Plot for Run ID: {run_id}\n(Seabed: {seabed_id}, Type: {synthetic_type})')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    output_path = os.path.join(output_dir, f'run_{run_id}_parity_plot.png')
    plt.savefig(output_path, dpi=300)
    print(f"    Saved parity plot to: {output_path}")
    # plt.show()
    plt.close()


def reconstruct(run_idx, cable_dict, seabed_dict):
    print(f"\n Starting Reconstruction Run {run_idx} ")

    # Generate a random load case to evaluate 
    seabed_x = np.arange(0, seabed_dict['profile_length'] + 1, 1)
    seabed_z = generate_seabed(np.random.randint(1, 1e6), seabed_dict['profile_length'] + 1)

    # Generate Ground Truth Profile for the entire seabed 
    print("Generating ground truth profile")
    true_solution = beam_on_elastic_foundation_bvp(
        x_a=seabed_x[0], y_a=seabed_z[0],
        x_b=seabed_x[-1], y_b=seabed_z[-1],
        slope_a=0, slope_b=0,
        T=cable_dict['residual_lay_tension'], EI=cable_dict['EI'],
        q_weight=cable_dict['submerged_weight'], k_foundation=seabed_dict['k_foundation'],
        seabed_x_coords=seabed_x, seabed_z_coords=seabed_z
    )
    if not true_solution:
        print("Could not generate ground truth solution. Skipping this run.")
        return
    true_cable_z = true_solution.sol(seabed_x)[0]
    
    # Load Model and Scalers 
    try:
        scalar_scaler = DataScaler.load(FEATURE_SCALER_PATH)
        profile_scaler = DataScaler.load(SEABED_SCALER_PATH)
        targets_scaler = DataScaler.load(TARGETS_SCALER_PATH)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model = InceptCurvesFiLM(**checkpoint['model_params']).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    except FileNotFoundError as e:
        print(f"Error loading files: {e}. Aborting.")
        return

    # Prepare for overlapping windowed reconstruction 
    
    if (seabed_dict['total_window_length'] - seabed_dict['centre_window_length']) % 2 != 0:
        print("Warning: seabed_dict['total_window_length'] - seabed_dict['centre_window_length'] must be even.")
        return
    buffer = (seabed_dict['total_window_length'] - seabed_dict['centre_window_length']) // 2

    # Define the start points of the overlapping central windows
    step_size = seabed_dict['centre_window_length'] - seabed_dict['overlap_length']
    if step_size <= 0:
        print(f"Error: Overlap ({seabed_dict['overlap_length']}m) must be smaller than the central window ({seabed_dict['centre_window_length']}m).")
        return
        
    centre_starts = []
    current_start = 0
    while current_start + seabed_dict['centre_window_length'] <= len(seabed_z) - 1:
        centre_starts.append(current_start)
        current_start += step_size
    
    centre_ends = [s + seabed_dict['centre_window_length'] for s in centre_starts]

    # Add buffer to each window for the CNN input
    window_starts = np.array(centre_starts) - buffer
    window_ends = np.array(centre_ends) + buffer

    # Calculate the required padding on the left and right sides
    pad_before = abs(min(window_starts)) if min(window_starts) < 0 else 0
    pad_after = (max(window_ends) - len(seabed_z)) if max(window_ends) > len(seabed_z) else 0

    # Create the padded Z array with the calculated asymmetrical padding
    padded_seabed_z = np.pad(seabed_z, pad_width=(pad_before, pad_after), mode='edge')

    # Create the corresponding X array, which is now guaranteed to have the same length
    padded_seabed_x = np.arange(-pad_before, len(seabed_z) + pad_after, 1)

    # Sanity check to prevent the error
    if len(padded_seabed_x) != len(padded_seabed_z):
        print("ERROR: Padded X and Z arrays do not match in length after correction")
        print(f"len(padded_seabed_x) = {len(padded_seabed_x)}")
        print(f"len(padded_seabed_z) = {len(padded_seabed_z)}")
        return # Skip this run if something is still wrong

    # Implement the overlapping window reconstruction process 

    # Get raw BC predictions for each overlapping window
    print("Getting raw predictions for each overlapping window")
    raw_bcs = []
    for i in range(len(centre_starts)):
        window_xs = np.arange(window_starts[i], window_ends[i], 1)
        window_zs = np.interp(window_xs, padded_seabed_x, padded_seabed_z)

        scaled_window_flat = torch.from_numpy(profile_scaler.transform(window_zs.reshape(1, -1))).float()
        scaled_window = scaled_window_flat.view(1, 1, -1).to(DEVICE)

        scalar_values = [cable_dict['EA'], cable_dict['EI'], cable_dict['submerged_weight'], cable_dict['residual_lay_tension']]
        scalar_array = np.array([scalar_values])
        scaled_scalars = torch.from_numpy(scalar_scaler.transform(scalar_array)).float().to(DEVICE)

        with torch.no_grad():
            predictions = model(scaled_window, scaled_scalars).cpu().numpy()
            predictions = targets_scaler.inverse_transform(predictions)
            raw_bcs.append(list(predictions[0]))

    # 5b. Create initial, overlapping reconstructions for each raw prediction
    print("Creating initial, overlapping reconstructions")
    initial_solutions = []
    for i in range(len(centre_starts)):
        z_a, z_b, slope_a, slope_b = raw_bcs[i]
        solution = beam_on_elastic_foundation_bvp(
            x_a=centre_starts[i], y_a=z_a, x_b=centre_ends[i], y_b=z_b,
            slope_a=slope_a, slope_b=slope_b,
            T=cable_dict['residual_lay_tension'],
            EI =cable_dict['EI'],
            q_weight=cable_dict['submerged_weight'],
            k_foundation=seabed_dict['k_foundation'],
            seabed_x_coords=padded_seabed_x, seabed_z_coords=padded_seabed_z
        )
        if solution:
            initial_solutions.append(solution)
        else:
            # If a solution fails, have to skip run
            print(f"BVP failed for initial segment {i}. Skipping this run.")
            break
    
    # Check if any BVP failed and we need to skip
    if len(initial_solutions) != len(centre_starts):
        print("\n Run failed - at least one BVP problem failed to converge")
        return

    # 5c. Define new junction points in the middle of overlaps and average the BCs
    print("Averaging BCs at new junctions within overlaps")
    junction_bcs = []
    for i in range(len(initial_solutions) - 1):
        # Define the new junction point in the middle of the overlapping region
        overlap_start = centre_starts[i+1]
        overlap_end = centre_ends[i]
        x_junction = (overlap_start + overlap_end) / 2.0
        
        # Get position and slope from the end of the first segment at the junction
        pos1, slope1, _, _ = initial_solutions[i].sol(x_junction)
        
        # Get position and slope from the start of the second segment at the junction
        pos2, slope2, _, _ = initial_solutions[i+1].sol(x_junction)
        
        # Average them to get the new, consistent boundary condition
        avg_z = (pos1 + pos2) / 2.0
        avg_slope = (slope1 + slope2) / 2.0
        
        junction_bcs.append({'x': x_junction, 'z': avg_z, 'slope': avg_slope})

    # 5d. Re-solve BVP for the final, non-overlapping segments using the new junction BCs
    print("Re-solving BVP for final, stitched segments")
    reconstructed_centres = []
    
    # Handle the first segment
    first_seg_start_bc = raw_bcs[0]
    first_seg_end_bc = junction_bcs[0]
    solution = beam_on_elastic_foundation_bvp(
        x_a=centre_starts[0], y_a=first_seg_start_bc[0],
        x_b=first_seg_end_bc['x'], y_b=first_seg_end_bc['z'],
        slope_a=first_seg_start_bc[2], slope_b=first_seg_end_bc['slope'],
        T=cable_dict['residual_lay_tension'],
        EI =cable_dict['EI'],
        q_weight=cable_dict['submerged_weight'],
        k_foundation=seabed_dict['k_foundation'],
        seabed_x_coords=padded_seabed_x, seabed_z_coords=padded_seabed_z
    )
    if solution:
        xs = np.linspace(centre_starts[0], first_seg_end_bc['x'], seabed_dict['profile_length'])
        reconstructed_centres.append({'xs': xs, 'zs': solution.sol(xs)[0]})

    # Handle the middle segments (from junction to junction)
    for i in range(len(junction_bcs) - 1):
        start_bc = junction_bcs[i]
        end_bc = junction_bcs[i+1]
        solution = beam_on_elastic_foundation_bvp(
            x_a=start_bc['x'], y_a=start_bc['z'],
            x_b=end_bc['x'], y_b=end_bc['z'],
            slope_a=start_bc['slope'], slope_b=end_bc['slope'],
            T=cable_dict['residual_lay_tension'],
            EI =cable_dict['EI'],
            q_weight=cable_dict['submerged_weight'],
            k_foundation=seabed_dict['k_foundation'],
            seabed_x_coords=padded_seabed_x, seabed_z_coords=padded_seabed_z
        )
        if solution:
            xs = np.linspace(start_bc['x'], end_bc['x'], seabed_dict['profile_length'])
            reconstructed_centres.append({'xs': xs, 'zs': solution.sol(xs)[0]})

    # Handle the last segment
    last_seg_start_bc = junction_bcs[-1]
    last_seg_end_bc = raw_bcs[-1]
    solution = beam_on_elastic_foundation_bvp(
        x_a=last_seg_start_bc['x'], y_a=last_seg_start_bc['z'],
        x_b=centre_ends[-1], y_b=last_seg_end_bc[1],
        slope_a=last_seg_start_bc['slope'], slope_b=last_seg_end_bc[3],
        T=cable_dict['residual_lay_tension'],
        EI =cable_dict['EI'],
        q_weight=cable_dict['submerged_weight'],
        k_foundation=seabed_dict['k_foundation'],
        seabed_x_coords=padded_seabed_x, seabed_z_coords=padded_seabed_z
    )
    if solution:
        xs = np.linspace(last_seg_start_bc['x'], centre_ends[-1], seabed_dict['profile_length'])
        reconstructed_centres.append({'xs': xs, 'zs': solution.sol(xs)[0]})

    #  6. Assemble the full predicted profile 
    print("Assembling full predicted profile")
    predicted_cable_z = np.full_like(seabed_z, np.nan)
    for segment in reconstructed_centres:
        start_idx = np.searchsorted(seabed_x, segment['xs'][0], side='left')
        end_idx = np.searchsorted(seabed_x, segment['xs'][-1], side='right')
        interp_z = np.interp(seabed_x[start_idx:end_idx], segment['xs'], segment['zs'])
        predicted_cable_z[start_idx:end_idx] = interp_z
    
    # Finish this section
    # Fill any remaining NaNs (usually at the very end) with the ground truth for a fair comparison
    nan_mask = np.isnan(predicted_cable_z)
    if np.any(nan_mask):
        print(f"  Warning: {np.sum(nan_mask)} points were not reconstructed. Filling with ground truth values for plotting.")
        # predicted_cable_z[nan_mask] = true_cable_z[nan_mask]
        predicted_cable_z[nan_mask] = 0

    #  7. Calculate Metrics and Generate Plots 
    print("Calculating metrics and generating plots")
    errors = predicted_cable_z - true_cable_z
    profile_rmse = np.sqrt(np.mean(errors**2))
    print(f"Profile displacement RMSE: {profile_rmse:.4f} m")

    # Create a mock run_data dict for plotting functions
    run_info = {'run_id': run_idx, 'seabed_id': 'Generated', 'synthetic': 'Reconstruction'}

    # Generate and save quantitative plots
    plot_error_histogram(errors, run_info, OUTPUT_DIR)
    plot_parity(true_cable_z, predicted_cable_z, run_info, OUTPUT_DIR)

    #  8. Generate and save the main qualitative profile plot 
    fig = go.Figure()
    
    # Plot the generated seabed
    fig.add_trace(go.Scattergl(
        x=seabed_x, y=seabed_z, mode='lines',
        name='Generated Seabed Profile', line=dict(color='grey', dash='dash', width=2)
    ))

    # Plot the ground truth cable profile
    fig.add_trace(go.Scattergl(
        x=seabed_x, y=true_cable_z, mode='lines',
        name='Ground Truth Cable', line=dict(color='limegreen', width=4, dash='solid')
    ))

    # Plot the initial, overlapping solutions for debugging 
    # These show the raw BVP solutions before blending
    initial_segment_colors = ['crimson', 'darkorange']
    for i, solution in enumerate(initial_solutions):
        # Define the x-range for this initial solution
        xs = np.linspace(centre_starts[i], centre_ends[i], seabed_dict['profile_length'])
        # Evaluate the solution to get the z-profile
        zs = solution.sol(xs)[0]
        fig.add_trace(go.Scattergl(
            x=xs, y=zs, mode='lines',
            name=f'Initial Segment {i}',
            # legendgroup='Initial Segments',
            line=dict(color=initial_segment_colors[i % 2], width=2, dash='dash'),
            opacity=0.7 # semi-transparent
        ))

    # Plot each final blended segment individually
    # These show the result after averaging at the junctions
    final_segment_colors = ['dodgerblue', 'cyan']
    for i, segment in enumerate(reconstructed_centres):
        fig.add_trace(go.Scattergl(
            x=segment['xs'], y=segment['zs'], mode='lines',
            name=f'Final Segment {i}', # Renamed for clarity
            # legendgroup='Final Segments',
            line=dict(color=final_segment_colors[i % 2], width=3),
            opacity=1.0
        ))

    fig.update_layout(
        title_text=f'Full Cable Profile Reconstruction for Run ID: {run_idx}',
        xaxis_title='Position (X) [m]', yaxis_title='Elevation (Z) [m]',
        template='plotly_dark'
    )
    
    output_path = os.path.join(OUTPUT_DIR, f'run_{run_idx}_profile.html')
    fig.write_html(output_path)
    print(f"Saved profile plot to: {output_path}")
    fig.show()


runs = [np.random.randint(1, 1e6) for x in range(5)]
cable_dict = {
    'EA': 9336351.79727941,
    'EI': 93363.5179727941,
    'submerged_weight': 71.14523286308311,
    'residual_lay_tension': 9033.41008549494
    }
seabed_dict = {
    'profile_length': 300,
    'total_window_length': 128,       # Base 2
    'centre_window_length': 74,       # Must be less than seabed_dict['total_window_length']
    'overlap_length': 10,     # Define the overlap in meters (e.g., 20m)
    'k_foundation': 1e5
}

for run_idx in runs:
    reconstruct(run_idx, cable_dict, seabed_dict)

print("\nAll reconstruction runs complete.")