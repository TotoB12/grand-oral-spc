# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime # Needed for formatting ETA
import numba # << ADDED for performance optimization (optional but recommended)

# --- [ Previous code Sections I to IV remain unchanged ] ---
# ... (Constants, Planet Data, Simulation Setup, Initial Conditions) ...
# Make sure PLANET_NAMES, SIMULATION_YEARS, TIME_STEP_DAYS, PERTURB_PLANET, etc. are defined here.

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
print("4. Plot INITIAL orbits (short duration) and FINAL positions to visualize divergence.") # << Updated Methodology
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

PROGRESS_UPDATE_INTERVAL = 1000

# --- NEW: Setup for Initial Orbit Plotting ---
INITIAL_ORBIT_PLOT_YEARS = 2 # How many years of the initial orbit to show
PLOT_SAMPLE_RATE = 100 # Store data more frequently initially if needed for smooth initial orbit plot
                       # Recalculate if needed based on TIME_STEP_DAYS
                       # Let's keep PLOT_SAMPLE_RATE relatively low (e.g., 100-1000)
                       # to have enough points for the initial orbit plot.
                       # If INITIAL_ORBIT_PLOT_YEARS is small, PLOT_SAMPLE_RATE can be smaller.

print("\nSimulation Configuration:")
print(f"  Planets Included: {', '.join(PLANET_NAMES)}")
print(f"  Simulation Duration: {SIMULATION_YEARS} years")
print(f"  Time Step: {TIME_STEP_DAYS} days")
print(f"  Perturbed Planet: {PERTURB_PLANET}")
print(f"  Perturbation Factor (x-pos): {PERTURB_FACTOR:.10f}")
print(f"  Plotting Initial Orbit Duration: {INITIAL_ORBIT_PLOT_YEARS} years") # << Info
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


# --- V. Physics and Integration (Numba functions) ---
# << Ensure the Numba functions calculate_accelerations and leapfrog_step are defined here >>
# << (Copied from previous version - unchanged) >>
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
    # In-place modification, no return needed within loop


# --- VI. Simulation Parameters ---
DT = TIME_STEP_DAYS * DAY
TOTAL_TIME = SIMULATION_YEARS * YEAR
NUM_STEPS = int(TOTAL_TIME / DT)

# Calculate how many *sampled* steps correspond to the initial plot duration
steps_per_year = YEAR / DT
initial_orbit_plot_steps = int(INITIAL_ORBIT_PLOT_YEARS * steps_per_year)
# Find the index in the *sampled* data array
initial_orbit_plot_samples = int(initial_orbit_plot_steps / PLOT_SAMPLE_RATE) + 1 # +1 to include start

print(f"\nSimulation Execution Details:")
print(f"  Time step (dt): {DT:.2f} seconds ({TIME_STEP_DAYS} days)")
print(f"  Total duration: {TOTAL_TIME:.2e} seconds ({SIMULATION_YEARS} years)")
print(f"  Number of steps: {NUM_STEPS}")
print(f"  Progress updates every {PROGRESS_UPDATE_INTERVAL} steps.")
print(f"  Plotting initial trajectory for {INITIAL_ORBIT_PLOT_YEARS} years ({initial_orbit_plot_steps} steps, {initial_orbit_plot_samples} sampled points).")
print(f"  Using Numba JIT compilation for speed: {'Yes' if numba.__version__ else 'No (Numba not found?)'}")
print("\n--- Starting Simulation ---")
print("(This will take several minutes, potentially less with Numba...)")


# --- VII. Run Simulation ---
plot_points_sim1 = [] # List to store sampled points for Sim 1 (Original)
# No need to store full trajectory for Sim 2 for plotting anymore

positions_sim1 = np.copy(initial_positions)
velocities_sim1 = np.copy(initial_velocities)
positions_sim2 = np.copy(initial_positions_perturbed)
velocities_sim2 = np.copy(initial_velocities_perturbed)

# Store initial state for plotting Sim 1's initial orbit
plot_points_sim1.append(np.copy(positions_sim1))

start_time = time.time()
# (Optional Numba compilation trigger can go here)

for step in range(NUM_STEPS):
    leapfrog_step(positions_sim1, velocities_sim1, masses, DT, G)
    leapfrog_step(positions_sim2, velocities_sim2, masses, DT, G)

    steps_done = step + 1
    # Store data point for Sim 1's trajectory plot periodically
    if steps_done % PLOT_SAMPLE_RATE == 0:
        # Only store if we are still within the initial orbit plotting phase
        # Reduces memory slightly, though not the main bottleneck anymore
        if steps_done <= initial_orbit_plot_steps + PLOT_SAMPLE_RATE: # Store a bit beyond needed steps just in case
             plot_points_sim1.append(np.copy(positions_sim1))
        # We don't need to store Sim 2 points for trajectory plotting

    # --- Progress Indicator (same as before) ---
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

# The final positions are stored in positions_sim1 and positions_sim2

end_time = time.time()
total_duration_seconds = end_time - start_time
print("\n\n--- Simulation Complete ---")
print(f"Total calculation time: {total_duration_seconds:.2f} seconds ({str(datetime.timedelta(seconds=int(total_duration_seconds)))}).")
print("-----------------------------------------------------")

# --- VIII. Post-Processing and Visualization ---
print("\nProcessing results for visualization...")

# Convert the collected plot points for Sim 1's initial orbit into a NumPy array
plot_trajectory_sim1_m = np.array(plot_points_sim1)
# Ensure we only take the number of samples needed for the initial orbit plot
num_samples_to_plot = min(initial_orbit_plot_samples, plot_trajectory_sim1_m.shape[0])
plot_trajectory_sim1_m = plot_trajectory_sim1_m[:num_samples_to_plot]

# Convert sampled initial trajectory from meters to AU
plot_trajectory_sim1_au = plot_trajectory_sim1_m / AU

# Final positions are the last calculated states
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

# --- Analysis Printout (same as before) ---
print(f"\nAnalysis of Results ({PERTURB_PLANET}):")
print(f"  Initial position difference: {pos_diff_m:.3f} meters")
print(f"  Final position (Sim 1 - Original): [{final_pos_pert_sim1[0]:.4f}, {final_pos_pert_sim1[1]:.4f}] AU")
print(f"  Final position (Sim 2 - Perturbed): [{final_pos_pert_sim2[0]:.4f}, {final_pos_pert_sim2[1]:.4f}] AU")
print(f"  Final separation distance: {final_distance_au:.4f} AU")
print(f"                         = {final_distance_km:,.0f} km")
print(f"\nConclusion: An initial difference of meters grew to ~{final_distance_km:,.0f} km over {SIMULATION_YEARS} years!")
print("This demonstrates the extreme sensitivity to initial conditions characteristic of chaos.")
print("-----------------------------------------------------")


# --- IX. Plotting (REVISED) ---
print("Generating plot (Initial Orbits + Final Positions)...")
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.5, color='gray')
ax.set_xlabel("Distance (AU)")
ax.set_ylabel("Distance (AU)")
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

plot_title = (f"Solar System Chaos Demo ({SIMULATION_YEARS} years)\n"
              f"Sensitivity to Initial Conditions ({PERTURB_PLANET} perturbed by factor {PERTURB_FACTOR:.10f})")
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

# --- Plot Initial Orbits (Sim 1 only) and Final Positions ---
legend_handles = [] # For creating a custom legend later

for i in range(num_bodies):
    planet_name = PLANET_NAMES[i]
    planet_color = colors[i]

    # 1. Plot Initial Orbit Segment (Sim 1 - Original)
    # Use a slightly thicker line for the initial orbit to make it clear
    line, = ax.plot(plot_trajectory_sim1_au[:, i, 0], plot_trajectory_sim1_au[:, i, 1],
                    '-', color=planet_color, lw=1.0, alpha=0.9, # Solid line for original initial orbit
                    label=f"{planet_name} Initial Orbit ({INITIAL_ORBIT_PLOT_YEARS} yrs)")
    if i == 0: # Add Sun label to legend handles
        legend_handles.append(line)
    elif i == perturb_index: # Special label for perturbed planet's initial orbit
        legend_handles.append(line) # Add to legend
    elif i > 0 : # Add other planets
         legend_handles.append(line)


    # 2. Plot Final Position (Sim 1 - Original)
    # Use large markers for final positions
    marker_size = 10 if i == 0 else 7 # Larger marker for Sun
    final_marker1, = ax.plot(final_pos_sim1_au[i, 0], final_pos_sim1_au[i, 1],
                             'o', color=planet_color, markersize=marker_size,
                             markeredgecolor='white', mew=1.0, # Clear edge
                             label=f"{planet_name} Final (Orig)")
    # Add only one representative marker type to legend later


    # 3. Plot Final Position (Sim 2 - Perturbed) - Only for planets
    if i > 0: # Don't plot perturbed Sun final position (it's virtually identical)
        final_marker2, = ax.plot(final_pos_sim2_au[i, 0], final_pos_sim2_au[i, 1],
                                 'X', color=planet_color, markersize=marker_size, # Use 'X' marker for perturbed
                                 markeredgecolor='cyan', mew=1.0, # Different edge color
                                 label=f"{planet_name} Final (Pert.)")
        # Add only one representative marker type to legend later


# --- Add Arrow for Perturbed Planet's Divergence ---
ax.annotate('', xy=final_pos_pert_sim2, xytext=final_pos_pert_sim1,
            arrowprops=dict(arrowstyle='<->', color='cyan', lw=2.0, shrinkA=5, shrinkB=5)) # Thicker arrow

# --- Create a Cleaner Legend ---
# Add representative markers to the legend handles
# We find the handles created by the plotting above
# Example: Add markers for Original (circle) and Perturbed (X) final positions
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='white', lw=1, label=f'Initial Orbit ({INITIAL_ORBIT_PLOT_YEARS} yrs)'),
    Line2D([0], [0], marker='o', color='gray', label='Final Pos (Orig)', # Use gray as neutral marker color
           markerfacecolor='gray', markersize=8, markeredgecolor='white', mew=1.0, linestyle='None'),
    Line2D([0], [0], marker='X', color='gray', label='Final Pos (Pert.)', # Use gray as neutral marker color
           markerfacecolor='gray', markersize=8, markeredgecolor='cyan', mew=1.0, linestyle='None'),
    Line2D([0], [0], color='cyan', lw=1.5, label=f'{PERTURB_PLANET} Final Separation')
]
# Add planet colors
for i in range(num_bodies):
     legend_elements.append(Line2D([0], [0], color=colors[i], lw=2, label=PLANET_NAMES[i]))


ax.legend(handles=legend_elements, loc='upper right', fontsize='small',
          facecolor='dimgray', edgecolor='white', labelcolor='white')

plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust rect to ensure suptitle fits

# --- Save Plot ---
plot_filename = "solar_system_chaos_demonstration_v2.png"
try:
    plt.savefig(plot_filename, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"\nPlot saved as '{plot_filename}'")
except Exception as e:
    print(f"\nError saving plot: {e}")

# --- Display Plot ---
print("Displaying plot window...")
plt.show()

print("\n--- End of Script ---")