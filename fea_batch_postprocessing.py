import pandas as pd
import numpy as np
import glob
import torch
import plotly.graph_objects as go
from simulation_utils import beam_on_elastic_foundation_bvp
import random
import time
import math
from scipy.optimize import minimize
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.metrics import mean_squared_error


'''
Script to create master batch files by collating individual data from each load case,
and then augmenting the dataset with a reflection in the y axis.
'''


def process_csv(filename):
    cols = ['run_id', 'seabed_id', 'youngs_modulus', 'EA', 'EI',
                'submerged_weight', 'residual_lay_tension',
                'cable_x', 'cable_z', 'seabed_x', 'seabed_z']
    df = pd.read_csv(filename, usecols=cols)
    metadata = df.iloc[0]

    seabed_x_raw = df['seabed_x'].dropna().to_numpy()
    seabed_z_raw = df['seabed_z'].dropna().to_numpy()
    cable_x_raw  = df['cable_x'].dropna().to_numpy()
    cable_z_raw  = df['cable_z'].dropna().to_numpy()

    global_x = seabed_x_raw[3:-3]       # Skip the first and last three nodes to remove outlier seabed elevations
    global_dx = global_x[1] - global_x[0]
    
    # Interpolate to get uniform grid
    seabed_z = np.interp(global_x, seabed_x_raw, seabed_z_raw)
    cable_z = np.interp(global_x, cable_x_raw, cable_z_raw)

    # Calculate targets for contact patch predictor
    is_contact = np.where(cable_z <= seabed_z, 1, 0)

    return {
        'run_id'                        : int(metadata['run_id']),
        'seabed_id'                     : int(metadata['seabed_id']),
        'youngs_modulus'                : metadata['youngs_modulus'],
        'EA'                            : metadata['EA'],
        'EI'                            : metadata['EI'],
        'submerged_weight'              : metadata['submerged_weight'],
        'residual_lay_tension'          : metadata['residual_lay_tension'],
        'global_x'                      : global_x.tolist(),
        'seabed_z'                      : seabed_z.tolist(),
        'cable_z'                       : cable_z.tolist(),
        'is_contact'                    : is_contact.tolist(),
        'global_dx'                     : global_dx,
        'synthetic'                     : 'original'
        }


def split_train_validation(df):

    shuffled_indices = torch.randperm(len(df), generator=torch.Generator().manual_seed(SEED)).numpy()
    split_point = int(VALIDATION_SPLIT * len(df))
    validation_indices = shuffled_indices[:split_point]
    df['split'] = 'training'
    df.loc[validation_indices, 'split'] = 'validation'
    print("\nDataset split summary:")
    print(f"{df['split'].value_counts()}\n")

    return(df)


def plot_run(run_id, master_filepath, windows_filepath):
    # Load the created Parquet file to verify and plot
    run_id = int(run_id)
    print(f"\nVerifying Run ID: {run_id}")
    df_master = pd.read_parquet(master_filepath, engine='pyarrow')
    df_windows = pd.read_parquet(windows_filepath, engine='pyarrow')

    # Find the specific run by its ID
    run_data_series = df_master[df_master['run_id'] == run_id]
    if run_data_series.empty:
        print(f"Error: Run ID {run_id} not found in the master file.")
        return
    
    # Filter windows for this run_id
    df_windows_run = df_windows[df_windows['run_id'] == run_id]
    if df_windows_run.empty:
        print(f"Warning: No windows found for Run ID {run_id} in the windows file.")
    
    run_synthetic_type = run_data_series['synthetic'].to_numpy()[0]
    if run_synthetic_type == 'original':
        # Find original run_id before augmentation
        max_original_run_id = df_master[df_master['synthetic'] == 'original']['run_id'].max()
        original_run_id = run_id % (max_original_run_id + 1)
        df_raw = pd.read_csv(f'{CSV_OUTPUT_FOLDER}\\load_case_{original_run_id}_results.csv')

    run = run_data_series.iloc[0]

    fig = go.Figure()

    if run_synthetic_type == 'original':
        # Add the Seabed Profile - FEA raw
        fig.add_trace(go.Scattergl(
            x=df_raw['seabed_x'],
            y=df_raw['seabed_z'],
            mode='lines',
            name='Seabed Profile FEA (Raw)',
            line=dict(color='red', dash='dash')
        ))

        # Add the FEA Cable Profile - FEA raw
        fig.add_trace(go.Scattergl(
            x=df_raw['cable_x'],
            y=df_raw['cable_z'],
            mode='lines',
            name='Cable Profile FEA (Raw)',
            line=dict(color='red', dash='dash')
        ))

    # Add the Seabed Profile - interpolated
    fig.add_trace(go.Scattergl(
        x=run['global_x'],
        y=run['seabed_z'],
        mode='lines',
        name='Seabed Profile FEA (Interpolated)',
        line=dict(color='grey', dash='dash')
    ))
    
    # Add the FEA Cable Profile - interpolated
    fig.add_trace(go.Scattergl(
        x=run['global_x'],
        y=run['cable_z'],
        mode='lines',
        name='Cable Profile FEA (Interpolated)',
        line=dict(color='dodgerblue', width=3)
    ))

    for i, window in df_windows_run.iterrows():
        # Add points for start and end of total window
        fig.add_trace(go.Scattergl(
            x=[window['window_start_x'], window['window_end_x']],
            y=[window['window_start_z'], window['window_end_z']],
            mode='markers',
            name=f"BVP Computed Windows {window['window_id']} Start/End",
        ))

        sol = beam_on_elastic_foundation_bvp(
            x_a=window['centre_start_x'],
            y_a=window['centre_start_z'],
            x_b=window['centre_end_x'],
            y_b=window['centre_end_z'],
            slope_a=window['fea_slope_start'],
            slope_b=window['fea_slope_end'],
            T=window['residual_lay_tension'],
            EI=window['EI'],
            q_weight=window['submerged_weight'],
            k_foundation=K_FOUNDATION,
            seabed_x_coords=np.array(run['global_x']),
            seabed_z_coords=np.array(run['seabed_z'])
        )

        if sol:
            fig.add_trace(go.Scattergl(
                x=sol.x,
                y=sol.y[0],
                mode='lines',
                name=f"BVP Computed Windows {window['window_id']}",
                line=dict(color='lime', width=2)
            ))
        else:
            print(f"BVP solver failed for run {run_id}, window {window['window_id']}")


    # Update layout to add titles, labels, and customize the look
    fig.update_layout(
        title_text=(
            f'Simulation Comparison for Run ID: {run_id} (Type: {run["synthetic"]})<br>'
            f'EA: {run["EA"]:.2e}, EI: {run["EI"]:.2e}, '
            f'Sub. Weight: {run["submerged_weight"]:.2f}, '
            f'Residual Lay Tension: {run["residual_lay_tension"]:.2f}'
        ),
        xaxis_title='Position (X) [m]',
        yaxis_title='Elevation (Z) [m]',
        template='plotly_dark' 
    )
    fig.show()


def process_windows(row, window_length, centre_length):
    row = row[1]
    # window_length = 2 ** int(np.log2(centre_length + 2*buffer_length))    # Readjust total window to get base 2 number of nodes

    buffer_length = (window_length - centre_length)/2
    cable_slope = np.gradient(row['cable_z'], row['global_x'])

    # # Create a temporary DataFrame for the current run
    # df_run = pd.DataFrame({
    #     'global_x': row['global_x'],
    #     'seabed_z': row['seabed_z'],
    #     'cable_z': row['cable_z'],
    #     'is_contact': row['is_contact']
    # })

    num_windows = int(len(row['global_x']) / centre_length) + 1

    # Split the model into windows
    window_starts = [0]
    window_ends = [window_length]
    for i in range(num_windows):
        if window_ends[-1] + centre_length > len(row['global_x']):
            pass
        else:
            window_starts.append(window_starts[-1] + centre_length)
            window_ends.append(window_ends[-1] + centre_length)

    # Generate window arrays
    window_results = []
    for i in range(len(window_starts)):
        window_start_x = window_starts[i]
        window_end_x = window_ends[i]

        window_xs = np.linspace(window_start_x, window_end_x, int(window_end_x - window_start_x))  # Assume 1 node per meter for now
        window_zs = np.interp(window_xs, row['global_x'], row['seabed_z'])

        # Get BCs at edges of each window from the FEA
        window_start_z = window_zs[0]
        window_end_z = window_zs[-1]
        centre_start_x = window_start_x + buffer_length
        centre_end_x = window_end_x - buffer_length
        centre_xs = np.arange(centre_start_x, centre_end_x, row['global_x'][1]-row['global_x'][0])
        centre_start_z = np.interp(centre_start_x, row['global_x'], row['cable_z'])
        centre_end_z = np.interp(centre_end_x, row['global_x'], row['cable_z'])
        fea_slope_start = np.interp(centre_start_x, row['global_x'], cable_slope)
        fea_slope_end = np.interp(centre_end_x, row['global_x'], cable_slope)

        # Append results to dictionary, will be converted to a dataframe at end
        window_results.append({
            'run_id': row['run_id'], 
            'window_id': i + 1,
            'seabed_id': row['seabed_id'],
            'youngs_modulus': row['youngs_modulus'], 
            'EA': row['EA'], 
            'EI': row['EI'],
            'submerged_weight': row['submerged_weight'], 
            'residual_lay_tension': row['residual_lay_tension'],
            'synthetic': row['synthetic'],
            'split': row['split'],
            'window_start_x': window_start_x,
            'window_end_x': window_end_x, 
            'window_start_z': window_start_z,
            'window_end_z': window_end_z,
            'centre_start_x': centre_start_x,
            'centre_end_x': centre_end_x,
            'centre_xs': centre_xs,
            'centre_start_z': centre_start_z,
            'centre_end_z': centre_end_z,
            'fea_slope_start': fea_slope_start,
            'fea_slope_end': fea_slope_end,
            'window_xs': window_xs.tolist(),
            'window_zs': window_zs.tolist(),
            'global_x': row['global_x'],
            'seabed_z': row['seabed_z']
        })

    return pd.DataFrame(window_results)


if __name__ == '__main__':
    MASTER_FILEPATH = 'master_batch_file.parquet'
    WINDOWS_FILEPATH = 'master_windows_file.parquet'
    CSV_OUTPUT_FOLDER = 'training_batch'
    VALIDATION_SPLIT = 0.3
    SEED = 123
    PROCESS_BATCH = True
    TOTAL_WINDOW_LENGTH = 128       # Base 2
    CENTRE_WINDOW_LENGTH = 74       # Must be less than TOTAL_WINDOW_LENGTH
    K_FOUNDATION = 1e5      # Assumption for seabed stiffness (N/m^2)

    if PROCESS_BATCH:
        csv_list = glob.glob('training_batch/*.csv')
        # csv_list = glob.glob('training_batch/*.csv')[:10]

        # Process master_df
        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(process_csv, csv_list)
        df_master = pd.DataFrame(results)
        print(f"Loaded {len(df_master)} original runs from CSV files.")

        # Split the data into training/validation
        df_master = split_train_validation(df_master)

        # Check the run_ids
        if df_master['run_id'].duplicated().any():
            print("Warning, duplicate run_id found in master dataframe.")
        elif df_master['run_id'].min() != 0 or df_master['run_id'].max() != len(df_master['run_id']) - 1:
            print("Warning, check minimum and maximum run_ids.")
            print(f"Min run_id: {df_master['run_id'].min()}, max run_id: {df_master['run_id'].max()}")

        # Augment the master dataset with synthetic data
        max_run_id = df_master['run_id'].max()
        df_master_aug = df_master.copy()
        df_master_aug['synthetic'] = 'x_flip'
        df_master_aug['run_id'] = df_master_aug['run_id'] + max_run_id
        df_master_aug['seabed_z'] = df_master_aug['seabed_z'].apply(np.flip)
        df_master_aug['cable_z'] = df_master_aug['cable_z'].apply(np.flip)
        df_master_aug['is_contact'] = df_master_aug['is_contact'].apply(np.flip)
        df_master_augmented = pd.concat([df_master, df_master_aug], ignore_index=True)
        # df = df.sort_values(by='run_id', ascending=False)
        df_master_augmented.to_parquet(MASTER_FILEPATH, engine='pyarrow', index=False)
        print(f"Master datastore saved with {len(df_master_augmented)} total entries saved to {MASTER_FILEPATH}")

        # Process windows
        start_time = time.time()
        results = []
        runs_to_process = list(df_master_augmented.iterrows())
        num_tasks = len(runs_to_process)
        num_workers = mp.cpu_count()
        chunksize = max(1, num_tasks // (num_workers * 4))

        print(f"Using {num_workers} workers (chunksize: {chunksize})")
        start_time = time.time()

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_windows, row, TOTAL_WINDOW_LENGTH, CENTRE_WINDOW_LENGTH): row
                for row in runs_to_process}
            # Collect results as they complete
            for i, future in enumerate(as_completed(futures), 1):
                result_df = future.result()
                results.append(result_df)
                if i % chunksize == 0 and i > 0:
                    elapsed_time = time.time() - start_time
                    estimated_remaining_time = (num_tasks - i) / (i / elapsed_time)
                    mins, secs = divmod(estimated_remaining_time, 60)
                    print(f"Processed windows for {i}/{num_tasks} runs")
                    print(f"Estimated time remaining: {int(mins)}m {int(secs)}s")
    
        df_windows = pd.concat(results, ignore_index=True)
        df_windows.to_parquet(WINDOWS_FILEPATH)

    # Plots for checking
    k = 2
    df_windows = pd.read_parquet(MASTER_FILEPATH)
    original_seabeds = set(df_windows[df_windows['synthetic'] == 'original']['seabed_id'].unique())
    flipped_seabeds = set(df_windows[df_windows['synthetic'] == 'x_flip']['seabed_id'].unique())
    common_seabed_ids = list(original_seabeds.intersection(flipped_seabeds))
    if len(common_seabed_ids) < k:
        print(f"Warning: Found only {len(common_seabed_ids)} augmented seabeds with pairs. Plotting all available.")
        k = len(common_seabed_ids)
    if k == 0:
        print("Could not find any original/flipped pairs to plot for verification.")
    else:
        random.seed(SEED)
        selected_seabed_ids = random.sample(common_seabed_ids, k=k)
        for seabed_id in selected_seabed_ids:
            original_run_id = df_windows[(df_windows['seabed_id'] == seabed_id) & (df_windows['synthetic'] == 'original')]['run_id'].iloc[0]
            flipped_run_id = df_windows[(df_windows['seabed_id'] == seabed_id) & (df_windows['synthetic'] == 'x_flip')]['run_id'].iloc[0]
            start_time = start_time = time.time()
            plot_run(original_run_id, MASTER_FILEPATH, WINDOWS_FILEPATH)
            elapsed_time = time.time() - start_time
            print(f'Time to solve BVP: {elapsed_time:.2f}sec')
            # plot_run(flipped_run_id, MASTER_FILEPATH)