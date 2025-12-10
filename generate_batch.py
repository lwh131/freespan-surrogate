import os
import pandas as pd
from scipy.stats import qmc
import time
from ansys.mapdl.core import launch_mapdl
from scipy.constants import g as GRAVITY
from scipy.constants import pi as PI
from simulation_utils import generate_seabed
import glob
import re
import psutil


def kill_ansys():
    target_process = "ANSYS.exe"
    killed = False

    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if proc.info['name'] == target_process:
                print(f"Killing {target_process} (PID: {proc.pid})")
                proc.terminate()
                proc.wait(timeout=5)
                killed = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if not killed:
        print(f"No running process named {target_process} found.")
    else:
        print(f"{target_process} terminated.")


LENGTH = 200
POISSON_RATIO = 0.3
RADIUS = 0.2
SUBMERGED_WEIGHT = 100
FRICTION = 0.3

CABLE_ELEMENT_LENGTH = RADIUS*4
N_CABLE_NODES = int(LENGTH/CABLE_ELEMENT_LENGTH)
N_CABLE_ELEMENTS = N_CABLE_NODES - 1
SEABED_START_LEFT = 2000001
SEABED_START_RIGHT = 3000001

#Auto set the run_id
run_id_list = []
for file in glob.glob('training_batch/load_case_*_results.csv'):
    match = re.search(r'load_case_(\d+)_results\.csv', file)
    if match:
        run_id_list.append(int(match.group(1)))
if len(run_id_list) == 0:
    START_RUN_ID = 1
else:
    START_RUN_ID = max(run_id_list) + 1

N_SAMPLES = 1e6

param_bounds = {
    'seabed_id'                 : [1, 1e8],
    'bending_stiffness_ei'      : [60e3, 120e3],
    'submerged_weight'          : [50, 150],
    'residual_lay_tension'      : [1000, 10000]
    }

# Generate the Latin Hypercube Samples
sampler = qmc.LatinHypercube(d=len(param_bounds))
sample = sampler.random(n=int(N_SAMPLES))

# Scale the samples to the defined bounds
scaled_samples = qmc.scale(sample, [b[0] for b in param_bounds.values()], [b[1] for b in param_bounds.values()])

# Create a directory for the simulation files
run_dir = 'training_batch'
if not os.path.exists(run_dir):
    os.makedirs(run_dir)

kill_ansys()
mapdl = launch_mapdl(run_location=run_dir, loglevel='ERROR', override=True, start_timeout=60)
print('PyMAPDL launched for batch processing.')

# Start the loop for each sample case
for index, params in enumerate(scaled_samples):
    run_id = START_RUN_ID + index
    seabed_id, bending_stiffness_ei, submerged_weight, residual_lay_tension = params
    seabed_id = int(seabed_id)
    seabed_profile = generate_seabed(seabed_id, LENGTH)

    start_time = time.time()
    print(f"Running simulation {run_id}/{int(START_RUN_ID + N_SAMPLES)}")
    print(time.ctime())
    print(f"Seabed ID = {seabed_id}")
    print(f"EI = {bending_stiffness_ei:.2e} N")
    print(f"Submerged weight = {submerged_weight:.0f} N/m")
    print(f"RLT = {residual_lay_tension:.0f} N")

    mapdl.clear()  # Clear the database for the new model
    jobname = f'load_case_{run_id}'
    mapdl.filname(jobname) # Set a unique jobname for each run
    mapdl.prep7() # Enter pre-processing

    # Set element type: BEAM188 is suitable for 2D/3D beam analysis
    mapdl.et(1, "BEAM188")  #1st set of elements of element type BEAM188

    # Set material properties based on the current sample
    # Calculate required youngs modulus to for the correct EI for a fixed radius
    youngs_modulus = bending_stiffness_ei / (PI / 4 * RADIUS**4)
    mapdl.mp("EX", 1, youngs_modulus) # Young's Modulus in Pa - ANSYS will apply it in all directions X/Y/Z
    mapdl.mp("PRXY", 1, POISSON_RATIO)
    # mapdl.mp("MU", 1, FRICTION)

    # Define cross-section: Circular with a diameter
    sec_num = 1
    mapdl.sectype(sec_num, "BEAM", "CSOLID") # Solid circular section
    mapdl.secdata(RADIUS) # Radius in meters

    #Real constants
    #1 is beam (secnum), 2 is line CONTA177, 3 is seabed TARGE177
    mapdl.r(2, "", "", -1e8, "", "", -0.5)     #R1, R2, FKN, FTOLN, ICONT, PINB           #R1 and R2 not needed as I am doing direct element generation, high FKN is stiffer, -ve PINB to specify absolute value
    mapdl.r(3)

    # Create geometry: make one node at the beginning and one node at the end then .fill the rest
    mapdl.n(1, 0, 0, 0)
    mapdl.n(N_CABLE_NODES, LENGTH, 0, 0)
    mapdl.fill(1, N_CABLE_NODES, N_CABLE_NODES-2)

    mapdl.type(1)
    for i in range(1, N_CABLE_NODES): # Loop from 1 to 100
        mapdl.e(i, i + 1)

    # Make the seabed
    # print(mapdl.mesh.nodes)
    seabed_seperation = RADIUS/2
    mapdl.n(SEABED_START_LEFT, -1, -1, -seabed_seperation)
    mapdl.n(SEABED_START_LEFT+LENGTH+1, LENGTH+2, -1, -seabed_seperation)
    mapdl.fill(SEABED_START_LEFT, SEABED_START_LEFT+LENGTH+1, LENGTH)

    mapdl.n(SEABED_START_RIGHT, -1, 1, -seabed_seperation)
    mapdl.n(SEABED_START_RIGHT+LENGTH+1, LENGTH+2, 1, -seabed_seperation)
    mapdl.fill(SEABED_START_RIGHT, SEABED_START_RIGHT+LENGTH+1, LENGTH)

    ###########################################################
    # Contact Model
    ###########################################################
    # Cable:
    mapdl.nsel("S", "NODE", "", 1, N_CABLE_NODES)
    mapdl.et(2, "CONTA177") # 2nd set of elements of element type CONTA177 
    mapdl.keyopt(2, 2, 1)   # Contact algo = penalty function
    mapdl.keyopt(2, 3, 1)   # KEYOPT(3) = 2 is contact pressure model - use with a negative FKN (FORCE/LENGTH)

    mapdl.keyopt(2, 12, 0)  # Behaviour of contact surface: standard
    mapdl.keyopt(2, 15, 3)  # Contact stabilization damping

    mapdl.type(2)
    mapdl.real(2)
    mapdl.esurf()   # Generate all contact elements

    # Seabed:
    mapdl.et(3, "TARGE170") 
    mapdl.keyopt(3, 12, 0)  # Select keyopts for standard contact algo and behaviour
    mapdl.keyopt(3, 2, 1) # Penalty method
    mapdl.nsel("S", "NODE", "", SEABED_START_LEFT, SEABED_START_LEFT+LENGTH+1)
    mapdl.nsel("A", "NODE", "", SEABED_START_RIGHT, SEABED_START_RIGHT+LENGTH+1)
    mapdl.type(3)
    mapdl.real(2)
    for i in range(0, int(LENGTH-1)):
        mapdl.e(SEABED_START_LEFT+i, SEABED_START_LEFT+1+i, SEABED_START_RIGHT+1+i, SEABED_START_RIGHT+i)

    mapdl.allsel()  # Select all
    mapdl.etlist()  # List currently defined element types
    mapdl.elist()   # Lists the elements and their attributes
    # mapdl.nplot()
    # mapdl.eplot()   # Plot currently selected elements

    mapdl.finish()       # Finish pre-processing
    mapdl.run("/solu")
    mapdl.antype("static")
    mapdl.nlgeom("ON")      # Activate large displacement analysis, essential for flexible cables
    mapdl.nropt("FULL")      # Use Full Newton-Raphson, often more stable
    mapdl.pred("ON")        # Activate the solution predictor for a better initial guess each substep
    mapdl.lnsrch("ON")      # Activate line search to prevent the solver from overshooting the solution
    mapdl.autots("ON") # Activate automatic time stepping
    # mapdl.stabilize("CONSTANT", "ENERGY", 0.1)
    mapdl.stabilize("DAMPING", 0.1)
    mapdl.cncheck("AUTO")    # Automatically check contact status and bisect if it changes
    mapdl.outres("ALL", "ALL")      # Tell Ansys to write results at the end of each substep for smooth animations

    ############################################################
    # Boundary Conditions and Loads 
    ############################################################
    # These are applied once and remain for all load steps.

    mapdl.nsel("S", "NODE", "", 1, N_CABLE_NODES)   # Select cable
    mapdl.d("ALL", "UY", 0)      # Fix in Y direction
    mapdl.d("ALL", "ROTX", 0)    # Remove torsion about beam axis
    mapdl.d("ALL", "ROTZ", 0)    # Fix rotation about Z

    mapdl.nsel("S", "NODE", "", 1, 1) # Select Node 1
    mapdl.d("ALL", "UX", 0)          # Fix X-translation
    # mapdl.d("ALL", "UZ", 0)          # Fix Z-translation
    
    mapdl.nsel("S", "NODE", "", SEABED_START_LEFT, SEABED_START_LEFT+LENGTH)
    mapdl.nsel("A", "NODE", "", SEABED_START_RIGHT, SEABED_START_RIGHT+LENGTH)
    mapdl.d("ALL", "UX", 0)
    mapdl.d("ALL", "UY", 0)
    mapdl.d("ALL", "UZ", 0)

    #LOAD STEP 1 - APPLT RLT
    mapdl.nsel("S", "NODE", "", N_CABLE_NODES, N_CABLE_NODES)
    mapdl.f("ALL", "FX", residual_lay_tension)
    mapdl.allsel()
    mapdl.lswrite(1)

    #LOAD STEP 2 - APPLY WEIGHT
    cable_area = PI * (RADIUS**2)
    required_density = submerged_weight / (cable_area * GRAVITY)
    mapdl.nsel("S", "NODE", "", 1, N_CABLE_NODES)
    mapdl.mp("DENS", 1, required_density)
    mapdl.acel(0, 0, GRAVITY) # Apply gravity in the +Z direction (Ansys convention)
    mapdl.allsel()
    mapdl.lswrite(2)
    mapdl.cncheck('detail')

    #LOAD STEP 3 - DEFORM SEABED
    for i in range(LENGTH):
        mapdl.nsel("S", "NODE", "", SEABED_START_LEFT+1+i)
        mapdl.nsel("A", "NODE", "", SEABED_START_RIGHT+1+i)
        mapdl.d("ALL", "UZ", seabed_profile[i])
    mapdl.nsubst(50, 100, 30)
    mapdl.allsel()
    mapdl.lswrite(3)

    mapdl.nsel('ALL') # Select everything before solving
    mapdl.lssolve(1, 3) # Solve all defined load steps

    ############################################################
    # Post processing
    ############################################################
    mapdl.post1() # Enter the general post-processor
    mapdl.set("LAST") # Select the last result set
    mapdl.save()

    # Create results dataframe
    df_cable = pd.DataFrame()
    df_seabed = pd.DataFrame()

    # Get all node numbers and their final coordinates from the mesh object
    all_node_num = mapdl.mesh.nnum

    # Create a boolean mask to select only the cable nodes (1 to N_CABLE_NODES)
    cable_mask = (all_node_num >= 1) & (all_node_num <= N_CABLE_NODES)
    # Filter node numbers and coordinates using the mask
    cable_node_nums = all_node_num[cable_mask]
    mapdl.nsel("S", "NODE", "", 1, N_CABLE_NODES)
    cable_nodes = mapdl.mesh.nodes
    cable_ux = mapdl.post_processing.nodal_values('U', 'X')
    cable_uz = mapdl.post_processing.nodal_values('U', 'Z')
    cable_uy = mapdl.post_processing.nodal_values('U', 'Y')
    seabed_gap = (mapdl.get_array(entity='NODE', item1='CONT', it1num='GAP'))[0:N_CABLE_NODES]
    seabed_pen = (mapdl.get_array(entity='NODE', item1='CONT', it1num='PENE'))[0:N_CABLE_NODES]
    df_cable = pd.DataFrame({
        'cable_node_no': cable_node_nums,
        'cable_x': cable_nodes[:, 0] + cable_ux,
        'cable_z': cable_nodes[:, 2] + cable_uz,
        'cable_y': cable_nodes[:, 1] + cable_uy,
        'seabed_gap': seabed_gap,
        'seabed_pen': seabed_pen
    })

    # Create boolean masks for both left and right seabed nodes
    seabed_mask = (all_node_num >= SEABED_START_LEFT) & (all_node_num <= SEABED_START_LEFT + LENGTH + 1)
    seabed_node_nums = all_node_num[seabed_mask]
    mapdl.nsel("S", "NODE", "", SEABED_START_LEFT, SEABED_START_LEFT + LENGTH + 1)
    seabed_nodes = mapdl.mesh.nodes
    seabed_ux = mapdl.post_processing.nodal_values('U', 'X')
    seabed_uz = mapdl.post_processing.nodal_values('U', 'Z')
    seabed_uy = mapdl.post_processing.nodal_values('U', 'Y')
    df_seabed = pd.DataFrame({
        'seabed_node_no': seabed_node_nums,
        'seabed_x': seabed_nodes[:, 0] + seabed_ux,
        'seabed_z': seabed_nodes[:, 2] + seabed_uz,
        'seabed_y': seabed_nodes[:, 1] + seabed_uy
    })

    # Get load case/metadata for simulation
    df_info = pd.DataFrame({
        'run_id'                : [run_id],
        'seabed_id'             : [seabed_id],
        'youngs_modulus'        : [youngs_modulus],
        'EA'                    : [youngs_modulus*PI*RADIUS**2],
        'EI'                    : [bending_stiffness_ei],
        'submerged_weight'      : [submerged_weight],
        'residual_lay_tension'  : [residual_lay_tension],
        'elem_length'           : [CABLE_ELEMENT_LENGTH]
    })

    df_results = pd.concat([df_info, df_cable, df_seabed], axis=1)
    df_filename = os.path.join(run_dir, f"{jobname}_results.csv")
    df_results.to_csv(df_filename, index=False)

    #Create image of final solution
    mapdl.vup(1, 'Z')         # Define the 'up' direction as Z for Window 1
    mapdl.view(1, 1, -1, 0)   # Window num, x, y, z defined the view line, scale is auto
    mapdl.pldisp(1)
    print(f'Analysis took {(time.time() - start_time) / 60:.2f} mins\n')
    print("---------------------------------------------------------------")
    mapdl.finish()

mapdl.exit()
print("\nPyMAPDL session closed.")
print("\n✅ Dataset generated.")
