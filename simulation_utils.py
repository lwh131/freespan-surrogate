import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import BSpline, LSQUnivariateSpline
from scipy.ndimage import gaussian_filter1d
import plotly.graph_objects as go
from perlin_noise import PerlinNoise
from scipy.integrate import solve_bvp
from scipy.interpolate import CubicSpline
from scipy.constants import g
from scipy.interpolate import interp1d


def beam_on_elastic_foundation_bvp(x_a, y_a, x_b, y_b, slope_a, slope_b, 
                                   T, EI, q_weight, k_foundation, 
                                   seabed_x_coords, seabed_z_coords):
    """Computes the profile of a cable section on an elastic foundation (seabed). (Winkler Model) """
    seabed_z_func = interp1d(seabed_x_coords, seabed_z_coords, kind='linear', fill_value="extrapolate")

    def ode_system(x, y):
        w, dw, M, V = y
        w_double_prime = (M / EI) * (1 + dw**2)**(1.5) if EI != 0 else 0
        seabed_z_at_x = seabed_z_func(x)
        deflection = seabed_z_at_x - w
        foundation_force = k_foundation * np.maximum(0, deflection)
        dV_dx = T * w_double_prime - q_weight + foundation_force
        return np.vstack((dw, w_double_prime, V, dV_dx))

    def boundary_conditions(ya, yb):
        return np.array([ya[0] - y_a, yb[0] - y_b, ya[1] - slope_a, yb[1] - slope_b])

    x_mesh = np.linspace(x_a, x_b, 201) # Increased mesh points for more complex profiles
    y_guess = np.zeros((4, x_mesh.size))
    y_guess[0] = seabed_z_func(x_mesh)
    y_guess[1] = np.gradient(y_guess[0], x_mesh)

    sol = solve_bvp(ode_system, boundary_conditions, x_mesh, y_guess, max_nodes=50000, tol=1e-4)
    return sol if sol.success else None


def fourth_order_beam_bvp(x_a, y_a, x_b, y_b, slope_a, slope_b, T, EI, q_weight):

    '''
    Computes the profile of a single hanging cable span using the geometrically
    exact 4th-order tensioned beam (beam-catenary) equation.

    This function solves the boundary value problem for a given set of start/end
    positions, start/end slopes, and known physical properties.

    Args:
        x_a (float): The horizontal position of the start of the span (End A).
        y_a (float): The vertical position of the start of the span (End A).
        x_b (float): The horizontal position of the end of the span (End B).
        y_b (float): The vertical position of the end of the span (End B).
        slope_a (float): The take-off slope (dy/dx) at End A.
        slope_b (float): The landing slope (dy/dx) at End B.
        T (float): The constant horizontal component of tension (N).
        EI (float): The bending stiffness of the cable (N·m²).
        q_weight (float): The distributed weight of the cable per unit length (N/m).

    Returns:
        scipy.integrate.OdeSolution: The solution object from the BVP solver.
        This object can be used to evaluate the cable shape and its derivatives
        at any point along the span. Returns None if the solver fails.
    '''

    # Define the System of First-Order ODEs 
    # State vector y = [w, w', M, V] where:
    # y[0] = w (vertical position)
    # y[1] = w' (slope)
    # y[2] = M (bending moment)
    # y[3] = V (shear force, dM/dx)
    def ode_system(x, y):
        w, dw, M, V = y
        
        # Avoid division by zero if EI is zero (pure catenary case)
        if EI == 0:
            w_double_prime = 0
        else:
            # From M = EI * w'' / (1 + (w')²)^(3/2), solve for w''
            # This is the key step to handle the geometric non-linearity.
            w_double_prime = (M / EI) * (1 + dw**2)**(1.5)

        # First order equations:
        # dw/dx = y[1]
        # dw'/dx = w'' = (M/EI)*(1+(w')^2)^(3/2)
        # dM/dx = V = y[3]
        # dV/dx = d²M/dx² = T*w'' - q_weight
        
        # Calculate dV/dx
        dV_dx = T * w_double_prime - q_weight
        
        return np.vstack((dw, w_double_prime, V, dV_dx))

    # Boundary Conditions 
    # The function needs to return an array of residuals that should be zero.
    def boundary_conditions(ya, yb):
        # ya: solution at the start of the interval (x_a)
        # yb: solution at the end of the interval (x_b)
        return np.array([
            ya[0] - y_a,      # w(a) = y_a
            yb[0] - y_b,      # w(b) = y_b
            ya[1] - slope_a,  # w'(a) = slope_a
            yb[1] - slope_b   # w'(b) = slope_b
        ])

    # Set up the Mesh and Initial Guess 
    x_mesh = np.linspace(x_a, x_b, 101)
    y_guess = np.zeros((4, x_mesh.size))
    
    try:
        # Cubic polynomial guess that matches the boundary conditions for the initial guess
        poly = CubicSpline(x=[x_a, x_b], y=[y_a, y_b], bc_type=((1, slope_a), (1, slope_b)))
        y_guess[0] = poly(x_mesh)      # Guess for w(x)
        y_guess[1] = poly(x_mesh, 1)   # Guess for w'(x)
    except (ValueError, np.linalg.LinAlgError):
        y_guess[0] = np.linspace(y_a, y_b, x_mesh.size)
        y_guess[1] = (y_b - y_a) / (x_b - x_a + 1e-9)

    # Solve the BVP
    sol = solve_bvp(ode_system, boundary_conditions, x_mesh, y_guess, verbose=0, max_nodes=5000)
    
    if not sol.success:
        print(f"BVP solver failed: {sol.message}")
        return None
        
    return sol


def plot_bvp_solution(sol, x_a, y_a, x_b, y_b):
    '''Helper function to visualize the solution using Plotly.'''
    
    # Generate a dense mesh for a smooth plot
    x_plot = np.linspace(x_a, x_b, 500)
    # The shape w(x) is the first component of the solution vector
    y_plot = sol.sol(x_plot)[0]

    fig = go.Figure()

    # Add the calculated cable shape
    fig.add_trace(go.Scatter(
        x=x_plot, y=y_plot, mode='lines', name='Cable Span Profile',
        line=dict(color='cyan', width=3)
    ))

    # Add markers for the start and end points
    fig.add_trace(go.Scatter(
        x=[x_a, x_b], y=[y_a, y_b], mode='markers', name='Endpoints',
        marker=dict(color='magenta', size=10, symbol='diamond')
    ))

    fig.update_layout(
        title='Solved Hanging Cable Span Profile',
        xaxis_title='Horizontal Position (m)',
        yaxis_title='Vertical Position (m)',
        template='plotly_dark',
        # Use equal aspect ratio to visualize angles correctly
        # yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    fig.show()


def elastic_catenary(x_start, z_start, x_end, z_end, tension_0, EA, weight_0, increments):
    '''
    Calculates the profile of an elastic caternary for a given tension on an
    unenven seabed when given coordinates of start and end positions, youngs
    modulus and weight.
    
    Parameters:
    tension_0   = (N) Horizontal component of tension in the cable
    EA          = (N) Axial rigidity EA of the cable, corresponds to E on Wikipedia E = kp, where k is spring stiffness, p is length
    weight_0    = (N/m) Weight per unit length of cable
    p           = (m) Natural unstretched length of a segment measured from the lowest point (vertex) of the catenary to the specific point of interest on the cable
    
    Returns
    x_array = (m) Array of length `increments` for x coordinates along the length
    z_array = (m) Array of length `increments` for z coordinates along the length
    '''
    a = tension_0 / weight_0
    approximate_length = np.hypot(x_end-x_start, z_end-z_start)

    def x_parametric(p, alpha):
        return a * np.arcsinh(p/a) + tension_0*p/EA + alpha
    
    def y_parametric(p, beta):
        return np.hypot(a, p) + (tension_0*p**2)/(2*EA*a) + beta

    def equations_to_solve(variables):
        p_start, p_end = variables

        eq1 = x_parametric(p_end, 0) - x_parametric(p_start, 0) + x_start - x_end
        eq2 = y_parametric(p_end, 0) - y_parametric(p_start, 0) + z_start - z_end

        return [eq1, eq2]
    
    def find_p_for_x(p): return x_parametric(p, alpha) - x

    # Apply boundary conditions to solve for alpha and beta: x_2 - x_1 = x_parametric(p_2) - x_parametric(p_1)
    solution = fsolve(equations_to_solve, [approximate_length/2, approximate_length/2])
    p_start, p_end = solution

    alpha = x_start - x_parametric(p_start, 0)
    beta = z_start - y_parametric(p_start, 0)

    # # Find the vertex
    # x_0 = x_parametric(0, alpha)
    # y_0 = y_parametric(0, beta)

    # Guess the value of p for a given x
    x_array = np.linspace(x_start, x_end, increments)
    z_array = []
    for x in x_array:
        p = (fsolve(find_p_for_x, approximate_length/2))[0]
        y = y_parametric(p, beta)
        z_array.append(y)

    return x_array, z_array


def calculate_spline(coefficients, x_uniform):
    '''
    Calculates a spline based on its coefficients, evaluating it on the given values in x_uniform.
    Returns both the spline and its 2nd and 3rd derivative. 
    '''
    coefficients = np.asarray(coefficients)
    x_uniform = np.asarray(x_uniform)

    degree = 3
    num_control_points = len(coefficients)

    # Determine the interval from x_uniform array
    x_min, x_max = x_uniform.min(), x_uniform.max()

    # Number of knots = number of control points + degree + 1
    num_knots = num_control_points + degree + 1

    # Create uniform knots in the interval [x_min, x_max]
    t_internal = np.linspace(x_min, x_max, num_knots - 2*degree)
    t = np.concatenate((
        np.repeat(t_internal[0], degree),
        t_internal,
        np.repeat(t_internal[-1], degree)
    ))

    # Construct BSpline with given knots, coefficients, and degree
    spline = BSpline(t, coefficients, degree)
    # Evaluate spline on the x_uniform points for loss or plotting
    y_eval = spline(x_uniform)
    # Exact 4th derivative
    d3_spline = spline.derivative(3)
    d3_eval = d3_spline(x_uniform)

    d2_spline = spline.derivative(2)
    d2_eval = d2_spline(x_uniform)

    # # Plot spline and control points
    # plt.plot(x_uniform, y_eval, label='Spline from coefficients')
    # # Map control points evenly spaced in [x_min, x_max] for plotting
    # control_x = np.linspace(x_min, x_max, num_control_points)
    # plt.scatter(control_x, coefficients, color='red', label='Control points')
    # plt.legend()
    # plt.xlabel('x')
    # plt.ylabel('Spline value')
    # plt.title('BSpline from coefficients and uniform knots')
    # plt.grid(True)
    # plt.show()

    return y_eval, d2_eval, d3_eval


def fit_spline(x, y, num_control_points):
    '''
    Fit a cubic spline with a specified number of coefficients to target data.

    Args:
        num_control_points (int): Number of spline coefficients (control points).
        target_data (tuple or list): Tuple or list of (x, y) arrays.

    Returns:
        np.ndarray: Array of fitted spline coefficients.
    '''
    degree = 3
    x = np.asarray(x)
    y = np.asarray(y)

    # Sort data by x, required for spline fitting
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    # Calculate number of internal knots
    # Total knots = len(coefficients) + degree + 1
    # Number internal knots = total knots - 2*(degree+1)
    num_internal_knots = num_control_points - (degree + 1)

    if num_internal_knots <= 0:
        raise ValueError("Number of coefficients too small for cubic spline.")

    # Select internal knots uniformly inside data range (exclude boundaries)
    internal_knots = np.linspace(x_sorted[0], x_sorted[-1], num_internal_knots + 2)[1:-1]

    # Fit LSQUnivariateSpline
    spline = LSQUnivariateSpline(x_sorted, y_sorted, internal_knots, k=degree)

    # Extract coefficients
    coefficients = spline.get_coeffs()

    return coefficients


def generate_seabed(seed, profile_length):

    noise_layers = {
        'layer1': {'octaves': 0.05, 'amplitude': 20, 'seed_offset': 0},
        'layer2': {'octaves': 0.1, 'amplitude': 10, 'seed_offset': 1},
        'layer3': {'octaves': 0.2, 'amplitude': 5, 'seed_offset': 2},
        'layer4': {'octaves': 0.4, 'amplitude': 2.5, 'seed_offset': 3},
        'layer5': {'octaves': 0.8, 'amplitude': 1.2, 'seed_offset': 4},
        'layer6': {'octaves': 1.6, 'amplitude': 0.6, 'seed_offset': 5},
        'layer7': {'octaves': 3.8, 'amplitude': 0.3, 'seed_offset': 6},
        'layer8': {'octaves': 7.7, 'amplitude': 0.1, 'seed_offset': 7},
        'layer9': {'octaves': 15.4, 'amplitude': 0.09, 'seed_offset': 8},
        'layer10': {'octaves': 30.7, 'amplitude': 0.04, 'seed_offset': 9},
        'layer11': {'octaves': 150, 'amplitude': 0.025, 'seed_offset': 10},
        'layer12': {'octaves': 20, 'amplitude': 0.1, 'seed_offset': 11}
    }
    
    # Initialize profile with zeros
    profile = np.zeros(profile_length)
    
    # Generate and combine all noise layers
    for layer_name, params in noise_layers.items():
        noise = PerlinNoise(octaves=params['octaves'], seed=seed + params['seed_offset'])
        layer_profile = [noise([i / profile_length]) * params['amplitude'] 
                         for i in range(profile_length)]
        profile += np.array(layer_profile)
    
    # Normalise to achieve ~2m range per 200m
    scaling_factor = (2 / (profile.max() - profile.min())) + (np.random.random_sample() - 0.5) / 2
    profile = profile * scaling_factor
    
    # Shift the profile to have a mean of 0 (optional)
    profile = profile - profile.mean()
    
    return profile


def check_contact(cable_x, cable_z):
    '''
    Returns a list of the implied contact points (cable_x) of a cable profile.
    '''
    cable_x = np.asarray(cable_x)
    cable_z = np.asarray(cable_z)

    # Calculate the 3rd derivative of cable profile
    d3_z = np.gradient(np.gradient((np.gradient(cable_z, cable_x)), cable_x), cable_x)

    crossings = []
    for i in range(len(d3_z) - 1):
        if d3_z[i] < 0 and d3_z[i+1] >= 0:
            x0, x1 = cable_x[i], cable_x[i+1]
            y0, y1 = d3_z[i], d3_z[i+1]
            if y1 != y0:
                x_zero = x0 - y0 * (x1 - x0) / (y1 - y0)
                crossings.append(x_zero)
    
    return crossings


def check_contact_exact(spline_coeffs, x_uniform):
    '''
    Returns a list of the implied contact points based on a set of spline coeffs.
    '''
    # Calculate the 3rd derivative of cable profile
    y_spline, d2_z, d3_z = calculate_spline(spline_coeffs, x_uniform)

    crossings = []
    for i in range(len(d3_z) - 1):
        if d3_z[i] < 0 and d3_z[i+1] >= 0:
            x0, x1 = x_uniform[i], x_uniform[i+1]
            y0, y1 = d3_z[i], d3_z[i+1]
            if y1 != y0:
                x_zero = x0 - y0 * (x1 - x0) / (y1 - y0)
                crossings.append(x_zero)
    
    return crossings


def plot_check_contact():
    # Test for the cable_physics_loss function
    for i in range(np.random.randint(4000)):
        try:
            df_fea_raw = pd.read_csv(f'training_batch\\load_case_{i}_results.csv')
        except FileNotFoundError:
            continue
        df_interp = pd.read_parquet('master_batch_file.parquet', engine='pyarrow')
        df_interp = df_interp[df_interp['run_id'] == i]

        plt.plot(df_fea_raw['cable_x'], df_fea_raw['cable_z'])
        plt.plot(df_fea_raw['seabed_x'], df_fea_raw['seabed_z'])

        crossings = check_contact(df_fea_raw['cable_x'], df_fea_raw['cable_z'])
        plt.scatter(crossings, [0] * len(crossings))
        plt.grid()
        plt.show()


def physics_loss(cable_x, cable_z, run_id):
    '''
    Calculate the physics loss for a given cable profile.
    '''

    # Check implied contact points against the real contact points
    df = pd.read_parquet('master_batch_file.parquet', engine='pyarrow')
    df = df.loc[df['run_id'] == run_id].iloc[0]
    real_contacts_mask = df['cable_z'] <= (df['seabed_z'] + 0.01)  #Make boolean mask
    real_contacts = df['global_x'][real_contacts_mask]    #Apply boolean mask

    model_output_coeffs = np.array([ 0.15497023,  0.22597918,  0.39297545,  1.0055761 ,  1.5222243 ,
        1.8249767 ,  1.73001   ,  1.4512713 ,  0.96938056,  0.7712547 ,
        1.0167372 ,  1.1975331 ,  1.0366666 ,  0.22726132, -0.7290964 ,
       -1.0763466 , -0.8094534 ,  0.078054  ,  1.6452825 ,  3.2279496 ,
        3.826483  ,  3.0433803 ,  1.591796  ,  0.590306  , -0.00771898,
       -0.4057983 , -0.5298633 , -0.37045   , -0.04602618,  0.34469727,
        0.8814956 ,  1.2252856 ,  1.5238067 ,  1.7151955 ,  1.3229266 ,
        0.79347676,  0.3523347 ,  0.23752727,  0.44202903,  0.50678587])

    # Find the implied contact points of the cable profile
    implied_contacts = check_contact_exact(model_output_coeffs, df['global_x'])

    df_fea_raw = pd.read_csv(f'training_batch\\load_case_{run_id}_results.csv')

    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=df_fea_raw['cable_x'],
        y=df_fea_raw['cable_z'],
        mode='lines',
        name='Cable FEA Raw'
        # line=dict(color='grey', wid='dash', width=2)
    ))

    fig.add_trace(go.Scattergl(
        x=df_fea_raw['seabed_x'],
        y=df_fea_raw['seabed_z'],
        mode='lines',
        name='Seabed FEA Raw'
        # line=dict(color='grey', dash='dash', width=2)
    ))

    fig.add_trace(go.Scattergl(
        x=implied_contacts,
        y=[0] * len(implied_contacts),
        mode='markers',
        name='Implied Contacts'
        # line=dict(color='grey', dash='dash', width=2)
    ))

    fig.add_trace(go.Scattergl(
        x=real_contacts,
        y=[0] * len(real_contacts),
        mode='markers',
        name='Real Contacts'
        # line=dict(color='grey', dash='dash', width=2)
    ))

    cnn_mlp_y, _, _ = calculate_spline(model_output_coeffs, df['global_x'])
    fig.add_trace(go.Scattergl(
        x=df['global_x'],
        y=cnn_mlp_y,
        mode='lines',
        name='CNN-MLP Cable'
        # line=dict(color='grey', dash='dash', width=2)
    ))

    fig.update_layout(
        title_text=f'Cable Profile Prediction for Run ID: {run_id})',
        xaxis_title='Position (X) [m]',
        yaxis_title='Elevation (Z) [m]',
        template='plotly_dark'
    )

    fig.show()


if __name__ == '__main__':
    # df = pd.read_parquet('master_batch_file.parquet', engine='pyarrow')
    # df = df.loc[df['run_id'] == 3].iloc[0]
    # physics_loss(df['global_x'], df['cable_z'], 3)

    # plot_check_contact()

    for i in range(100000):
        fig, ax = plt.subplots(figsize=(15, 10))
        seabed = generate_seabed(np.random.randint(1, 1e6), 200)
        ax.plot(seabed, linestyle='-', marker='o', markersize=2)
        ax.set_ylim(np.mean(seabed) - 2, np.mean(seabed) + 2)
        ax.grid()
        plt.show()

    # Define Cable Physical Properties
    # cable_diameter = 0.2  # meters
    # steel_density = 7850  # kg/m^3
    # E_modulus = 200e9     # Pascals (steel)
    # gravity = 9.81        # m/s^2

    # I_moment = np.pi * (cable_diameter**4) / 64
    # A_cross_section = np.pi * (cable_diameter**2) / 4
    
    # EI_stiffness = E_modulus * I_moment
    # q_cable_weight = steel_density * g * A_cross_section
    
    # Define Span and Boundary Conditions 
    # print(" Solving Scenario 1: Shallow Catenary ")
    # x_a = 0
    # y_a = -25
    # x_b = 100
    # y_b = 15
    # slope_a = 0.2
    # slope_b = -0.2
    # tension = 10000

    # span_solution_1 = fourth_order_beam_bvp(x_a, y_a, x_b, y_b, slope_a, slope_b, tension, EI_stiffness, q_cable_weight)
    
    # if span_solution_1:
    #     plot_bvp_solution(span_solution_1, x_a, y_a, x_b, y_b)