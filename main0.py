# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime # Needed for formatting ETA
import numba # << ADDED for performance optimization (optional but recommended)
import os  # << ADDED to create output directory

print("-----------------------------------------------------")
print(" Grand Oral SPC - Solar System Chaos Demonstration ")
print("-----------------------------------------------------")
print("Objective: Demonstrate the chaotic nature of the Solar System")
print("           by showing extreme sensitivity to initial conditions.")
print("\nMethodology:")
print("1. Simulate the gravitational interactions of planets (N-body problem).")
print("2. Run TWO simulations in parallel:")
print("   - Simulation 1: Based on standard initial conditions.")
print("   - Simulation 2: Identical, EXCEPT for a minuscule perturbation")
print("     applied to the starting position of one planet (Mercury).")
print("3. Integrate the orbits over a long period (10,000 years).")
print("4. Plot:")
print("   - The detailed trajectories for the FIRST 10 years.")
print("   - The FINAL positions after 10,000 years to visualize divergence.")
print("\nKey Concept: Deterministic Chaos")
print("   The laws of physics (Newton's gravity) are deterministic.")
print("   However, due to complex interactions (N >= 3 bodies),")
print("   even infinitesimally small differences in starting conditions")
print("   grow exponentially over long timescales, making precise long-term")
print("   prediction impossible in practice.")
print("-----------------------------------------------------")

# --- I. Constants ---
G = 6.67430e-11     # Gravitational constant (m^3 kg^-1 s^-2)
AU = 1.495978707e11 # Astronomical Unit (meters) - More precise value
DAY = 60 * 60 * 24  # Seconds in a day
YEAR = 365.25 * DAY # Seconds in a year (Julian year)

# --- II. Planet Data ---
# Format: [Mass (kg), Semi-major axis (AU), Orbital Period (years - for reference), Color]
# Using simplified initial conditions: circular orbits, starting aligned on x-axis.
M_SUN = 1.9885e30 # Mass of Sun (kg)
planet_raw_data = {
    # Name      Mass (kg)   Dist (AU) Period(yr) Color
    'Sun':     [M_SUN,      0.0,      0.0,       'yellow'],
    'Mercury': [3.3011e23,  0.387,    0.241,     'gray'],
    'Venus':   [4.8675e24,  0.723,    0.615,     'orange'],
    'Earth':   [5.9724e24,  1.0,      1.0,       'blue'],
    'Mars':    [6.4171e23,  1.524,    1.881,     'red'],
    # Optional: Add Jupiter for more interactions, but increases calculation time significantly
    # 'Jupiter': [1.8982e27,  5.204,    11.86,     'tan'],
}

# --- III. Simulation Setup ---
PLANET_NAMES = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars'] # Focus on inner planets where chaos is more pronounced
SIMULATION_YEARS = 10000 # Simulate for 10,000 years to see large divergence
TIME_STEP_DAYS = 0.1     # Small time step for accuracy
PLOT_LIMIT_AU = 2.0      # Plotting range in AU for inner planets

# << NEW: Define how many years of trajectory lines to plot >>
PLOT_TRAJECTORY_YEARS = 10

PERTURB_PLANET = 'Mercury'
PERTURB_FACTOR = 1.00000001 # 1 + 1e-8

# --- Progress Update Frequency ---
PROGRESS_UPDATE_INTERVAL = 1000 # Update progress every 1000 steps

# --- Output Setup ---
OUTPUT_DIR = "chaos_simulation_output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
PLOT_FILENAME = os.path.join(OUTPUT_DIR, "solar_system_chaos_demonstration.png")


print("\nSimulation Configuration:")
print(f"  Planets Included: {', '.join(PLANET_NAMES)}")
print(f"  Total Simulation Duration: {SIMULATION_YEARS} years")
print(f"  Time Step: {TIME_STEP_DAYS} days")
print(f"  Perturbed Planet: {PERTURB_PLANET}")
print(f"  Perturbation Factor (x-pos): {PERTURB_FACTOR:.10f}")
print(f"  Plotting detailed trajectory for first: {PLOT_TRAJECTORY_YEARS} years") # << Info
print(f"  Plotting final positions after: {SIMULATION_YEARS} years") # << Info
print(f"  Output will be saved in: '{OUTPUT_DIR}/'")
print("-----------------------------------------------------")

# --- IV. Process Initial Conditions ---
masses = []
initial_positions_m = []  # Store in meters
initial_velocities_mps = [] # Store in m/s
colors = []

print("Setting up initial conditions (simplified circular orbits)...")
for name in PLANET_NAMES:
    data = planet_raw_data[name]
    mass = data[0]
    dist_au = data[1]
    color = data[3]

    masses.append(mass)
    colors.append(color)

    dist_m = dist_au * AU
    initial_positions_m.append([dist_m, 0.0])

    if name == 'Sun' or dist_m == 0:
        initial_velocities_mps.append([0.0, 0.0])
    else:
        velocity_mag_mps = np.sqrt(G * M_SUN / dist_m)
        initial_velocities_mps.append([0.0, velocity_mag_mps])

masses = np.array(masses)
initial_positions = np.array(initial_positions_m)
initial_velocities = np.array(initial_velocities_mps)
num_bodies = len(PLANET_NAMES)

# --- Create Perturbed State ---
initial_positions_perturbed = np.copy(initial_positions)
initial_velocities_perturbed = np.copy(initial_velocities)

perturb_index = PLANET_NAMES.index(PERTURB_PLANET)
original_pos = initial_positions_perturbed[perturb_index, 0]
initial_positions_perturbed[perturb_index, 0] *= PERTURB_FACTOR
perturbed_pos = initial_positions_perturbed[perturb_index, 0]
pos_diff_m = perturbed_pos - original_pos

print(f"\nPerturbing initial state:")
print(f"  Applying perturbation to {PERTURB_PLANET} (index {perturb_index}).")
print(f"  Original x-position: {original_pos / AU:.10f} AU")
print(f"  Perturbed x-position: {perturbed_pos / AU:.10f} AU")
print(f"  Initial difference: {pos_diff_m:.3f} meters ({pos_diff_m / 1000:.3f} km)")
print("-----------------------------------------------------")

# --- V. Physics and Integration (Numba optimized) ---
@numba.njit(fastmath=True, cache=True)
def calculate_accelerations(positions, masses, G_const):
    num_bodies_local = masses.shape[0]
    accelerations = np.zeros_like(positions, dtype=np.float64)
    for i in range(num_bodies_local):
        for j in range(num_bodies_local):
            if i == j: continue
            r_vector = positions[j] - positions[i]
            dist_sq = np.sum(r_vector**2)
            if dist_sq > 1e-6:
                dist = np.sqrt(dist_sq)
                acceleration_magnitude = G_const * masses[j] / dist_sq
                accelerations[i] += acceleration_magnitude * (r_vector / dist)
    return accelerations

@numba.njit(fastmath=True, cache=True)
def leapfrog_step(positions, velocities, masses, dt, G_const):
    initial_accelerations = calculate_accelerations(positions, masses, G_const)
    velocities += 0.5 * initial_accelerations * dt
    positions += velocities * dt
    final_accelerations = calculate_accelerations(positions, masses, G_const)
    velocities += 0.5 * final_accelerations * dt
    # Modification happens in-place

# --- VI. Simulation Parameters ---
DT = TIME_STEP_DAYS * DAY  # Time step in seconds
TOTAL_TIME = SIMULATION_YEARS * YEAR # Total simulation time in seconds
NUM_STEPS = int(TOTAL_TIME / DT)

# << NEW: Calculate steps for the trajectory plotting period >>
TIME_TO_PLOT_TRAJ = PLOT_TRAJECTORY_YEARS * YEAR
STEPS_TO_PLOT_TRAJ = int(TIME_TO_PLOT_TRAJ / DT)
# Ensure we don't try to plot more steps than we simulate
STEPS_TO_PLOT_TRAJ = min(STEPS_TO_PLOT_TRAJ, NUM_STEPS)

print(f"\nSimulation Execution Details:")
print(f"  Time step (dt): {DT:.2f} seconds ({TIME_STEP_DAYS} days)")
print(f"  Total duration: {TOTAL_TIME:.2e} seconds ({SIMULATION_YEARS} years)")
print(f"  Number of steps: {NUM_STEPS}")
print(f"  Steps to store for trajectory plot: {STEPS_TO_PLOT_TRAJ} (First {PLOT_TRAJECTORY_YEARS} years)") # << Info
print(f"  Progress updates every {PROGRESS_UPDATE_INTERVAL} steps.")
print(f"  Using Numba JIT compilation for speed: {'Yes' if numba.__version__ else 'No (Numba not found?)'}")
print("\n--- Starting Simulation ---")
print("(This will take several minutes, potentially less with Numba...)")

# --- VII. Run Simulation ---

# << MODIFIED: Use lists to store trajectory points ONLY for the first PLOT_TRAJECTORY_YEARS >>
plot_points_sim1 = []
plot_points_sim2 = []

# Initialize simulation states
positions_sim1 = np.copy(initial_positions)
velocities_sim1 = np.copy(initial_velocities)
positions_sim2 = np.copy(initial_positions_perturbed)
velocities_sim2 = np.copy(initial_velocities_perturbed)

# Store initial states for plotting
plot_points_sim1.append(np.copy(positions_sim1))
plot_points_sim2.append(np.copy(positions_sim2))

start_time = time.time()
last_update_time = start_time

for step in range(NUM_STEPS):
    # Update Simulation 1 & 2 (in-place modification)
    leapfrog_step(positions_sim1, velocities_sim1, masses, DT, G)
    leapfrog_step(positions_sim2, velocities_sim2, masses, DT, G)

    # --- Store data point for plotting trajectory lines IF within the defined period ---
    # We store data for step 0 already, so check step+1
    if (step + 1) < STEPS_TO_PLOT_TRAJ:
        plot_points_sim1.append(np.copy(positions_sim1))
        plot_points_sim2.append(np.copy(positions_sim2))
    # --- Special case: Store the very last point of the plotting period ---
    elif (step + 1) == STEPS_TO_PLOT_TRAJ:
         plot_points_sim1.append(np.copy(positions_sim1))
         plot_points_sim2.append(np.copy(positions_sim2))
         print(f"\n   (Finished storing trajectory data for first {PLOT_TRAJECTORY_YEARS} years at step {step+1})") # Info message

    # --- Progress Indicator (remains the same) ---
    steps_done = step + 1
    if steps_done % PROGRESS_UPDATE_INTERVAL == 0 or steps_done == NUM_STEPS:
        current_time = time.time()
        elapsed_total = current_time - start_time
        progress_pct = steps_done / NUM_STEPS * 100
        current_sim_year = (steps_done * DT) / YEAR

        eta_str = "Calculating..."
        if progress_pct > 0.1 and elapsed_total > 1.0:
            time_per_step = elapsed_total / steps_done
            remaining_steps = NUM_STEPS - steps_done
            eta_seconds = time_per_step * remaining_steps
            eta_formatted = str(datetime.timedelta(seconds=int(eta_seconds)))
            eta_str = f"ETA: {eta_formatted}"

        elapsed_formatted = str(datetime.timedelta(seconds=int(elapsed_total)))
        status_line = (f"  Progress: {progress_pct:5.1f}% | Year: {current_sim_year:6.0f}/{SIMULATION_YEARS} "
                       f"| Elapsed: {elapsed_formatted} | {eta_str}")
        print(status_line.ljust(80), end='\r')
        last_update_time = current_time

# Note: The final positions (positions_sim1, positions_sim2) now hold the state
# after the FULL SIMULATION_YEARS duration.

end_time = time.time()
total_duration_seconds = end_time - start_time
print("\n\n--- Simulation Complete ---")
print(f"Total calculation time: {total_duration_seconds:.2f} seconds ({str(datetime.timedelta(seconds=int(total_duration_seconds)))}).")
print("-----------------------------------------------------")

# --- VIII. Post-Processing and Visualization ---
print("\nProcessing results for visualization...")

# Convert the collected trajectory points (first N years) into NumPy arrays
plot_trajectory_sim1_m = np.array(plot_points_sim1)
plot_trajectory_sim2_m = np.array(plot_points_sim2)

# Convert sampled trajectories from meters to AU
plot_trajectory_sim1_au = plot_trajectory_sim1_m / AU
plot_trajectory_sim2_au = plot_trajectory_sim2_m / AU

# The final positions are the state after the FULL simulation
final_pos_sim1_m = positions_sim1
final_pos_sim2_m = positions_sim2

# Convert final positions to AU
final_pos_sim1_au = final_pos_sim1_m / AU
final_pos_sim2_au = final_pos_sim2_m / AU

# Calculate the final difference for the perturbed planet (after full simulation)
final_pos_pert_sim1 = final_pos_sim1_au[perturb_index]
final_pos_pert_sim2 = final_pos_sim2_au[perturb_index]
final_difference_vector = final_pos_pert_sim2 - final_pos_pert_sim1
final_distance_au = np.linalg.norm(final_difference_vector)
final_distance_km = final_distance_au * AU / 1000

print(f"\nAnalysis of Results ({PERTURB_PLANET}):")
print(f"  Initial position difference: {pos_diff_m:.3f} meters")
print(f"  Position after {PLOT_TRAJECTORY_YEARS} years (Sim 1): [{plot_trajectory_sim1_au[-1, perturb_index, 0]:.4f}, {plot_trajectory_sim1_au[-1, perturb_index, 1]:.4f}] AU") # Pos at end of plotted trajectory
print(f"  Position after {PLOT_TRAJECTORY_YEARS} years (Sim 2): [{plot_trajectory_sim2_au[-1, perturb_index, 0]:.4f}, {plot_trajectory_sim2_au[-1, perturb_index, 1]:.4f}] AU")
print(f"  FINAL position after {SIMULATION_YEARS} years (Sim 1): [{final_pos_pert_sim1[0]:.4f}, {final_pos_pert_sim1[1]:.4f}] AU")
print(f"  FINAL position after {SIMULATION_YEARS} years (Sim 2): [{final_pos_pert_sim2[0]:.4f}, {final_pos_pert_sim2[1]:.4f}] AU")
print(f"  FINAL separation distance after {SIMULATION_YEARS} years: {final_distance_au:.4f} AU")
print(f"                                               = {final_distance_km:,.0f} km")
print(f"\nConclusion: An initial difference of meters grew to ~{final_distance_km:,.0f} km over {SIMULATION_YEARS} years!")
print("The plot shows the initial similarity (first 10 years) and the vastly different final positions.")
print("-----------------------------------------------------")

# --- IX. Plotting ---
print(f"Generating plot (Trajectories for first {PLOT_TRAJECTORY_YEARS} yrs, Final positions for {SIMULATION_YEARS} yrs)...")
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.5, color='gray')
ax.set_xlabel("Distance (AU)")
ax.set_ylabel("Distance (AU)")
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

# Plot title and info
plot_title = (f"Solar System Chaos: Trajectories ({PLOT_TRAJECTORY_YEARS} yrs) & Final Positions ({SIMULATION_YEARS} yrs)\n"
              f"Sensitivity to Initial Conditions ({PERTURB_PLANET} perturbed by factor {PERTURB_FACTOR:.10f})")
fig.suptitle(plot_title, fontsize=14, color='white')
ax.set_title(f"Final Separation of {PERTURB_PLANET} after {SIMULATION_YEARS} yrs: {final_distance_au:.3f} AU ({final_distance_km:,.0f} km)",
             fontsize=10, color='cyan')

ax.set_xlim(-PLOT_LIMIT_AU, PLOT_LIMIT_AU)
ax.set_ylim(-PLOT_LIMIT_AU, PLOT_LIMIT_AU)
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.yaxis.label.set_color('white')
ax.xaxis.label.set_color('white')
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.spines['left'].set_color('white')

# Store legend handles and labels
handles = []
labels = []

# Plot trajectories (first N years) and final positions (after full sim)
for i in range(num_bodies):
    # --- Plot Trajectory Lines (First N Years) ---
    # Simulation 1 (Original) - Solid lines
    line1, = ax.plot(plot_trajectory_sim1_au[:, i, 0], plot_trajectory_sim1_au[:, i, 1],
                     '-', color=colors[i], lw=1.0, alpha=0.9, # Slightly thicker lines might be ok now
                     label=f"_{PLANET_NAMES[i]} Traj (Orig)") # Use underscore prefix to hide from auto-legend initially

    # Simulation 2 (Perturbed) - Dashed lines (only for planets)
    if i > 0:
        line2, = ax.plot(plot_trajectory_sim2_au[:, i, 0], plot_trajectory_sim2_au[:, i, 1],
                         '--', color=colors[i], lw=1.0, alpha=0.8,
                         label=f"_{PLANET_NAMES[i]} Traj (Pert)") # Underscore prefix

    # --- Plot Final Position Markers (After Full Simulation) ---
    # Simulation 1 (Original) - Larger marker
    marker1 = ax.plot(final_pos_sim1_au[i, 0], final_pos_sim1_au[i, 1],
                      'o', color=colors[i], markersize=9 if i==0 else 6, markeredgecolor='white', mew=0.5,
                      label=f"{PLANET_NAMES[i]} Final (Orig)")[0] # Capture the marker object

    # Simulation 2 (Perturbed) - Smaller marker (only for planets)
    marker2 = None
    if i > 0:
        marker2 = ax.plot(final_pos_sim2_au[i, 0], final_pos_sim2_au[i, 1],
                          's', color=colors[i], markersize=4, alpha=0.9,
                          label=f"{PLANET_NAMES[i]} Final (Pert)")[0] # Capture the marker object

    # --- Add to custom legend ---
    # Add only one entry per planet, representing its final state markers
    if i == 0: # Sun
         handles.append(marker1)
         labels.append(f"{PLANET_NAMES[i]} Final")
    elif i == perturb_index: # Perturbed Planet
         handles.append(marker1)
         labels.append(f"{PLANET_NAMES[i]} Final (Orig)")
         if marker2:
             handles.append(marker2)
             labels.append(f"{PLANET_NAMES[i]} Final (Pert)")
    else: # Other planets
         handles.append(marker1)
         labels.append(f"{PLANET_NAMES[i]} Final (Orig)")
         if marker2:
             # Optionally add perturbed markers for non-perturbed planets too if desired
             # handles.append(marker2)
             # labels.append(f"{PLANET_NAMES[i]} Final (Pert)")
             pass


# Add an arrow pointing between the FINAL positions of the perturbed planet
ax.annotate('', xy=final_pos_pert_sim2, xytext=final_pos_pert_sim1,
            arrowprops=dict(arrowstyle='<->', color='cyan', lw=1.5, shrinkA=5, shrinkB=5))

# Add custom legend showing only final position markers
ax.legend(handles, labels, loc='upper right', fontsize='small', facecolor='dimgray', edgecolor='white', labelcolor='white', title="Final Positions")
plt.tight_layout(rect=[0, 0.03, 1, 0.92]) # Adjust layout slightly more for longer title

# Save the figure before showing it
try:
    plt.savefig(PLOT_FILENAME, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"\nPlot saved as '{PLOT_FILENAME}'")
except Exception as e:
    print(f"\nError saving plot: {e}")

# Display the plot window
print("Displaying plot window...")
plt.show()

print("\n--- End of Script ---")