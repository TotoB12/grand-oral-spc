# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime # Needed for formatting ETA
import numba # << ADDED for performance optimization (optional but recommended)

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
print("3. Integrate the orbits over a long period (thousands of years).")
print("4. Plot the final trajectories of both simulations to visualize divergence.")
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
SIMULATION_YEARS = 10000 # << INCREASED DURATION! Simulate for 10,000 years
TIME_STEP_DAYS = 0.1     # << SMALLER TIME STEP for better accuracy over long duration
PLOT_LIMIT_AU = 2.0      # Plotting range in AU for inner planets

PERTURB_PLANET = 'Mercury'
# Apply a tiny perturbation (1 part in 100 million) to the initial x-position
# This represents the unavoidable uncertainty in measurements.
PERTURB_FACTOR = 1.00000001 # 1 + 1e-8

# --- Progress Update Frequency ---
# Update progress indicator every N steps to avoid excessive printing/calculation overhead
PROGRESS_UPDATE_INTERVAL = 1000 # Update progress every 1000 steps

# --- MEMORY FIX: Trajectory Downsampling ---
# Instead of storing every step, store position data only every N steps for plotting
# This drastically reduces memory usage.
PLOT_SAMPLE_RATE = 1000 # Store data for plotting every 1000 simulation steps

print("\nSimulation Configuration:")
print(f"  Planets Included: {', '.join(PLANET_NAMES)}")
print(f"  Simulation Duration: {SIMULATION_YEARS} years")
print(f"  Time Step: {TIME_STEP_DAYS} days")
print(f"  Perturbed Planet: {PERTURB_PLANET}")
print(f"  Perturbation Factor (x-pos): {PERTURB_FACTOR:.10f}")
print(f"  Plotting Trajectory Sample Rate: 1 point per {PLOT_SAMPLE_RATE} steps") # << Info
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

    # Position: Start all planets along the positive x-axis
    dist_m = dist_au * AU
    initial_positions_m.append([dist_m, 0.0])

    # Velocity: Calculate for a circular orbit (v = sqrt(G*M_sun / r))
    # Direction is along the positive y-axis
    if name == 'Sun' or dist_m == 0: # Sun starts at rest at the origin
        initial_velocities_mps.append([0.0, 0.0])
    else:
        # Use M_SUN for velocity calculation (approximation: sun's mass dominates)
        velocity_mag_mps = np.sqrt(G * M_SUN / dist_m)
        initial_velocities_mps.append([0.0, velocity_mag_mps])

masses = np.array(masses)
initial_positions = np.array(initial_positions_m)
initial_velocities = np.array(initial_velocities_mps)
num_bodies = len(PLANET_NAMES)

# --- Create Perturbed State ---
initial_positions_perturbed = np.copy(initial_positions)
initial_velocities_perturbed = np.copy(initial_velocities) # Velocities start identical

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

# --- V. Physics and Integration ---

# << SPEED OPTIMIZATION: Added Numba JIT compilation >>
# The @numba.njit decorator compiles this function to machine code for speed.
# 'fastmath=True' allows some potentially unsafe floating point optimizations,
# often giving extra speed but use with caution if extreme precision is paramount.
# 'cache=True' saves the compiled version to disk for faster subsequent runs.
@numba.njit(fastmath=True, cache=True)
def calculate_accelerations(positions, masses, G_const):
    """Calculates the acceleration on each body due to gravitational forces
       from all other bodies. Uses N-body calculation."""
    num_bodies_local = masses.shape[0] # Use shape[0] inside Numba function
    accelerations = np.zeros_like(positions, dtype=np.float64) # Explicit float64

    # Use Numba's prange for potential parallel execution of the outer loop
    # Requires Numba >= 0.50 and careful handling of shared variables if any
    # For simple accumulation like this, it's often safe.
    # Remove 'parallel=True' if you encounter issues or don't have suitable hardware.
    # for i in numba.prange(num_bodies_local): # << Potential parallelization
    for i in range(num_bodies_local): # Standard sequential loop
        for j in range(num_bodies_local):
            if i == j:
                continue # No self-gravity

            # Vector from body i to body j
            r_vector = positions[j] - positions[i]
            # Numba handles np.sum efficiently
            dist_sq = np.sum(r_vector**2)

            # Avoid division by zero and excessive force at very close range
            # Add a small softening factor (e.g., 1 meter squared) - optional
            # softening_sq = 1.0
            # dist = np.sqrt(dist_sq + softening_sq)
            # if dist > 1e-3: # Check avoids sqrt(0)

            # Check if distance is practically zero before sqrt
            if dist_sq > 1e-6: # Avoid sqrt(small number) issues and division by zero
                dist = np.sqrt(dist_sq)
                # Force magnitude: G * m_i * m_j / dist^2
                # Acceleration magnitude on i: G * m_j / dist^2
                acceleration_magnitude = G_const * masses[j] / dist_sq
                # Acceleration vector on i: magnitude * unit_vector (r_vector / dist)
                accelerations[i] += acceleration_magnitude * (r_vector / dist)
            # else:
                # Handle potential close encounters if needed (e.g., log warning)
                # Note: Numba doesn't support print directly in nopython mode easily
                pass

    return accelerations

# << SPEED OPTIMIZATION: Added Numba JIT compilation >>
@numba.njit(fastmath=True, cache=True)
def leapfrog_step(positions, velocities, masses, dt, G_const):
    """Performs one step of the Leapfrog integration method (Kick-Drift-Kick)."""
    # Kick 1 (Update velocities by half step)
    # Pass G explicitly as Numba compiles functions separately
    initial_accelerations = calculate_accelerations(positions, masses, G_const)
    velocities += 0.5 * initial_accelerations * dt
    # Drift (Update positions by full step using new velocities)
    positions += velocities * dt
    # Kick 2 (Update velocities by another half step using new positions)
    final_accelerations = calculate_accelerations(positions, masses, G_const)
    velocities += 0.5 * final_accelerations * dt
    # Numba functions should return the modified arrays explicitly if needed elsewhere,
    # but here modification happens in-place, which is fine.
    # return positions, velocities # Not strictly needed if modified in-place

# --- VI. Simulation Parameters ---
DT = TIME_STEP_DAYS * DAY  # Time step in seconds
TOTAL_TIME = SIMULATION_YEARS * YEAR # Total simulation time in seconds
NUM_STEPS = int(TOTAL_TIME / DT)

print(f"\nSimulation Execution Details:")
print(f"  Time step (dt): {DT:.2f} seconds ({TIME_STEP_DAYS} days)")
print(f"  Total duration: {TOTAL_TIME:.2e} seconds ({SIMULATION_YEARS} years)")
print(f"  Number of steps: {NUM_STEPS}")
print(f"  Progress updates every {PROGRESS_UPDATE_INTERVAL} steps.")
print(f"  Using Numba JIT compilation for speed: {'Yes' if numba.__version__ else 'No (Numba not found?)'}") # Check if Numba imported
print("\n--- Starting Simulation ---")
print("(This will take several minutes, potentially less with Numba...)")

# --- VII. Run Simulation ---

# << MEMORY FIX: Use lists to store downsampled trajectory points >>
# Initialize lists to hold the position data for plotting
plot_points_sim1 = []
plot_points_sim2 = []

# Initialize simulation states (use copies!)
positions_sim1 = np.copy(initial_positions)
velocities_sim1 = np.copy(initial_velocities)
positions_sim2 = np.copy(initial_positions_perturbed)
velocities_sim2 = np.copy(initial_velocities_perturbed)

# Store initial states for plotting
plot_points_sim1.append(np.copy(positions_sim1))
plot_points_sim2.append(np.copy(positions_sim2))

start_time = time.time()
last_update_time = start_time

# --- Call Numba functions once to trigger compilation (optional, avoids slight delay on first step) ---
# try:
#     _ = leapfrog_step(positions_sim1, velocities_sim1, masses, DT, G)
#     print("Numba functions compiled successfully.")
# except Exception as e:
#     print(f"Numba compilation might have failed: {e}")
# # Reset positions/velocities if needed after test call (or just let the loop start)
# positions_sim1 = np.copy(initial_positions)
# velocities_sim1 = np.copy(initial_velocities)
# ----------------------------------------------------------------------------------------------------


for step in range(NUM_STEPS):
    # Update Simulation 1 (Modify arrays in-place)
    leapfrog_step(positions_sim1, velocities_sim1, masses, DT, G)

    # Update Simulation 2 (Modify arrays in-place)
    leapfrog_step(positions_sim2, velocities_sim2, masses, DT, G)

    # --- MEMORY FIX: Store data point only periodically ---
    steps_done = step + 1
    if steps_done % PLOT_SAMPLE_RATE == 0:
        plot_points_sim1.append(np.copy(positions_sim1))
        plot_points_sim2.append(np.copy(positions_sim2))

    # --- Progress Indicator ---
    if steps_done % PROGRESS_UPDATE_INTERVAL == 0 or steps_done == NUM_STEPS:
        current_time = time.time()
        elapsed_total = current_time - start_time

        progress_pct = steps_done / NUM_STEPS * 100
        current_sim_year = (steps_done * DT) / YEAR

        eta_str = "Calculating..."
        if progress_pct > 0.1 and elapsed_total > 1.0: # Calculate ETA once some time has passed
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

# --- Ensure the very last state is captured for plotting if not sampled ---
if NUM_STEPS % PLOT_SAMPLE_RATE != 0:
     plot_points_sim1.append(np.copy(positions_sim1))
     plot_points_sim2.append(np.copy(positions_sim2))


end_time = time.time()
total_duration_seconds = end_time - start_time
print("\n\n--- Simulation Complete ---") # Newlines to move past the progress indicator
print(f"Total calculation time: {total_duration_seconds:.2f} seconds ({str(datetime.timedelta(seconds=int(total_duration_seconds)))}).")
print("-----------------------------------------------------")

# --- VIII. Post-Processing and Visualization ---
print("\nProcessing results for visualization...")

# << MEMORY FIX: Convert the collected plot points (lists) into NumPy arrays >>
# Shape will be (num_plot_points, num_bodies, 2) for each simulation
plot_trajectory_sim1_m = np.array(plot_points_sim1)
plot_trajectory_sim2_m = np.array(plot_points_sim2)

# Convert sampled trajectories from meters to Astronomical Units (AU) for plotting
plot_trajectory_sim1_au = plot_trajectory_sim1_m / AU
plot_trajectory_sim2_au = plot_trajectory_sim2_m / AU

# The final positions are simply the last calculated states
final_pos_sim1_m = positions_sim1
final_pos_sim2_m = positions_sim2

# Convert final positions to AU
final_pos_sim1_au = final_pos_sim1_m / AU
final_pos_sim2_au = final_pos_sim2_m / AU

# Calculate the final difference for the perturbed planet
final_pos_pert_sim1 = final_pos_sim1_au[perturb_index]
final_pos_pert_sim2 = final_pos_sim2_au[perturb_index]
final_difference_vector = final_pos_pert_sim2 - final_pos_pert_sim1
final_distance_au = np.linalg.norm(final_difference_vector)
final_distance_km = final_distance_au * AU / 1000

print(f"\nAnalysis of Results ({PERTURB_PLANET}):")
print(f"  Initial position difference: {pos_diff_m:.3f} meters")
print(f"  Final position (Sim 1): [{final_pos_pert_sim1[0]:.4f}, {final_pos_pert_sim1[1]:.4f}] AU")
print(f"  Final position (Sim 2): [{final_pos_pert_sim2[0]:.4f}, {final_pos_pert_sim2[1]:.4f}] AU")
print(f"  Final separation distance: {final_distance_au:.4f} AU")
print(f"                         = {final_distance_km:,.0f} km")
print(f"\nConclusion: An initial difference of meters grew to ~{final_distance_km:,.0f} km over {SIMULATION_YEARS} years!")
print("This demonstrates the extreme sensitivity to initial conditions characteristic of chaos.")
print("-----------------------------------------------------")

# --- IX. Plotting ---
print("Generating plot...")
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.5, color='gray')
ax.set_xlabel("Distance (AU)")
ax.set_ylabel("Distance (AU)")
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

# Plot title and info
plot_title = (f"Solar System Chaos Demo ({SIMULATION_YEARS} years)\n"
              f"Sensitivity to Initial Conditions ({PERTURB_PLANET} perturbed by factor {PERTURB_FACTOR:.10f})") # Show factor precision
fig.suptitle(plot_title, fontsize=14, color='white')
ax.set_title(f"Final Separation of {PERTURB_PLANET}: {final_distance_au:.3f} AU ({final_distance_km:,.0f} km)",
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

# Plot trajectories (using downsampled data) and final positions
for i in range(num_bodies):
    # Simulation 1 (Original) - Solid lines, larger final marker
    # << PLOTTING FIX: Use the downsampled trajectory data >>
    ax.plot(plot_trajectory_sim1_au[:, i, 0], plot_trajectory_sim1_au[:, i, 1],
            '-', color=colors[i], lw=0.5, alpha=0.7, # Thinner line for potentially many points
            label=f"{PLANET_NAMES[i]} (Orig)" if i != perturb_index else f"{PLANET_NAMES[i]} (Orig - Solid)")

    # Plot final position using the accurately stored final state
    ax.plot(final_pos_sim1_au[i, 0], final_pos_sim1_au[i, 1],
            'o', color=colors[i], markersize=9 if i==0 else 6, markeredgecolor='white', mew=0.5)

    # Simulation 2 (Perturbed) - Dashed lines, smaller final marker (only for planets)
    if i > 0: # Don't plot perturbed Sun trajectory (it barely moves anyway)
        # << PLOTTING FIX: Use the downsampled trajectory data >>
        ax.plot(plot_trajectory_sim2_au[:, i, 0], plot_trajectory_sim2_au[:, i, 1],
                '--', color=colors[i], lw=0.5, alpha=0.6, # Thinner line
                label=f"{PLANET_NAMES[i]} (Perturbed)" if i == perturb_index else None) # Label only perturbed planet

        # Plot final position using the accurately stored final state
        ax.plot(final_pos_sim2_au[i, 0], final_pos_sim2_au[i, 1],
                's', color=colors[i], markersize=4, alpha=0.9) # Use squares for perturbed final pos

# Add an arrow pointing between the final positions of the perturbed planet
ax.annotate('', xy=final_pos_pert_sim2, xytext=final_pos_pert_sim1,
            arrowprops=dict(arrowstyle='<->', color='cyan', lw=1.5, shrinkA=5, shrinkB=5)) # Add shrink to avoid overlap

ax.legend(loc='upper right', fontsize='small', facecolor='dimgray', edgecolor='white', labelcolor='white')
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

# << SAVE PLOT: Save the figure before showing it >>
plot_filename = "solar_system_chaos_demonstration.png"
try:
    plt.savefig(plot_filename, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"\nPlot saved as '{plot_filename}'")
except Exception as e:
    print(f"\nError saving plot: {e}")

# << Display the plot >>
print("Displaying plot window...")
plt.show()

print("\n--- End of Script ---")