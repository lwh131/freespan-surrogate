# Freespan Surrogate Model for Subsea Cables

This is an experimental learning project where I explore how to automate finite element simulations (FEA) simulations using ANSYS Mechanical APDL via PyMAPDL, generate structured datasets, and train a convolutional neural network (CNN) to predict the freespans of subsea pipelines or cables. I started this project to explore different ways of combining mechanical simulation data with machine learning models.

## Background and Motivation

In subsea engineering, on-bottom roughness analysis is a standard FEA workflow used to assess freespan formation when a pipeline or cable is laid on a rough seabed. These analyses can often require:

- non-linear pipe-soil contact modelling
- large deformations
- repeated analyses over many load cases

This project explores whether a fast surrogate can approximate freespan behaviour well enough to support:

- concept stage subsea route optimisation
- interactive design tools
- freespan hotspot visualisation
- scenario or route screening before running detailed FEA

The conceptual use case would be a subsea layout planning tool where:

- A user draws a proposed cable or pipeline route on a map
- The surrogate quickly predicts freespan statistics (number, length, height, elevation)
- Span statistics and hotspot visualisations are displayed instantly

The work is ongoing and currently focused on validating the modelling pipeline and improving the surrogate architecture.

## Project Status

- Complete: Automated FEA data generation pipeline
- Complete: Data processing, augmentation, scaling
- Complete: Initial CNN training and evaluation
- In Progress: Architecture experiments, improving prediction smoothness and accuracy
- Planned: Non-linear bending stiffness and sequential loading conditions

## Workflow
1. FEA Generation
   - Cable/pipeline properties (bending stiffness, weight, tension) and seabed bathymetry profiles are varied. ANSYS MAPDL runs non-linear on-bottom roughness analyses via PyMAPDL.
2. Training Data Creation
   - For each simulation, short snapshots of model are extracted. Cable elevation and slope are extracted for training inputs.
3. Physics Model
   - Solve the initial value problem for a beam on an elastic foundation (Winkler) to find the cable elevations and slopes that match the FEA case
4. Training
   - A PyTorch CNN is trained to output cable elevations and cable slopes from seabed profiles
5. Runtime Evaluation
   - The input seabed divided into overlapping windows. Each window is fed into the CNN to predict local cable shape.
6. Smoothing
   - Adjacent window predictions are not continuous, so the mean cable elevation and slope are extracted at the centre of each window and passed into the CNN again, enforcing a smooth cable profile. 

## File Structure

- `generate_batch.py`: Controls Ansys MAPDL to run FEA simulations and generate the raw .csv training data
- `fea_batch_postprocessing.py`: Collates raw data, performs data augmentation, and creates the master parquet training file
- `training.py`: Contains the primary training
- `models.py`: Defines the PyTorch models
- `torch_utils.py`: Contains helper functions
- `simulation_utils.py`: Contains helper functions
- `evaluation.py`: Evaluates the model's performance against the ground truth data and generates comparison plots
- `early_stopping.py`: A utility class for early stopping with checkpoint saving
- `preprocessing.py`: Contains the DataScaler utility class for standardising input features

## Viewing Results

- Run `reconstruction.py` to see the full workflow where a generated seabed is split into windows, input into the CNN and the outputs are pieced together to form a continuous cable shape prediction.
- Alternatively, download the `.html` plots in `/reconstruction_plots` and open them in the browser

<img width="2032" height="967" alt="newplot (28)" src="https://github.com/user-attachments/assets/7c3ec5c8-d3c4-46e1-93e7-a69155521b6c" />

<img width="2032" height="967" alt="newplot (27)" src="https://github.com/user-attachments/assets/58fc9a24-c964-4889-b7c3-583334cd21cb" />

<img width="2032" height="967" alt="newplot (26)" src="https://github.com/user-attachments/assets/3936f85c-d054-493f-b71f-41efd3606f94" />

## Future Work
1. Add user drawn cable routes, span statistics and span hotspot visualisations
2. Enforcing physics constraints in the final predicted cable shape
3. More realistic seabed generation and parameter ranges
4. Longer seabed models with finer mesh
5. Add bilinear bending stiffness and kinematic hardening for flexible pipes
6. Add sequential loading conditions
7. Automated hyperparameter optimisation

## Future Reading
1. Suitability of chosen model architecture
2. Other model architectures
3. Surrogate model uncertainty quantification
