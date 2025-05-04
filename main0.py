# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time

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
    # Optional: Add Jupiter for more interactions, but increases calculation time
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

print("\nSimulation Configuration:")
print(f"  Planets Included: {', '.join(PLANET_NAMES)}")
print(f"  Simulation Duration: {SIMULATION_YEARS} years")
print(f"  Time Step: {TIME_STEP_DAYS} days")
print(f"  Perturbed Planet: {PERTURB_PLANET}")
print(f"  Perturbation Factor (x-pos): {PERTURB_FACTOR:.10f}")
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

def calculate_accelerations(positions, masses):
    """Calculates the acceleration on each body due to gravitational forces
       from all other bodies. Uses N-body calculation."""
    num_bodies = len(masses)
    accelerations = np.zeros_like(positions, dtype=float) # Use float for precision
    for i in range(num_bodies):
        for j in range(num_bodies):
            if i == j:
                continue # No self-gravity

            # Vector from body i to body j
            r_vector = positions[j] - positions[i]
            dist_sq = np.sum(r_vector**2)
            dist = np.sqrt(dist_sq)

            # Avoid division by zero if bodies are too close (shouldn't happen in stable sim)
            # Add a small softening factor epsilon^2 to dist_sq if needed, but better with accurate steps
            if dist > 1e-3: # Check for very small distance (e.g., 1 meter)
                # Force magnitude: G * m_i * m_j / dist^2
                # Acceleration magnitude on i: G * m_j / dist^2
                acceleration_magnitude = G * masses[j] / dist_sq
                # Acceleration vector on i: magnitude * unit_vector (r_vector / dist)
                accelerations[i] += acceleration_magnitude * (r_vector / dist)
            # else:
                # Handle close encounters or collisions if necessary (not implemented here)
                # print(f"Warning: Close encounter between body {i} and {j} at distance {dist} m")
                # pass

    return accelerations

def leapfrog_step(positions, velocities, masses, dt):
    """Performs one step of the Leapfrog integration method (Kick-Drift-Kick)."""
    # Kick 1 (Update velocities by half step)
    initial_accelerations = calculate_accelerations(positions, masses)
    velocities += 0.5 * initial_accelerations * dt
    # Drift (Update positions by full step using new velocities)
    positions += velocities * dt
    # Kick 2 (Update velocities by another half step using new positions)
    final_accelerations = calculate_accelerations(positions, masses)
    velocities += 0.5 * final_accelerations * dt
    return positions, velocities

# --- VI. Simulation Parameters ---
DT = TIME_STEP_DAYS * DAY  # Time step in seconds
TOTAL_TIME = SIMULATION_YEARS * YEAR # Total simulation time in seconds
NUM_STEPS = int(TOTAL_TIME / DT)

print(f"\nSimulation Execution Details:")
print(f"  Time step (dt): {DT:.2f} seconds ({TIME_STEP_DAYS} days)")
print(f"  Total duration: {TOTAL_TIME:.2e} seconds ({SIMULATION_YEARS} years)")
print(f"  Number of steps: {NUM_STEPS}")
print("\n--- Starting Simulation ---")
print("(This will take several minutes...)")

# --- VII. Run Simulation ---
# Allocate memory for trajectories (store every N steps to save memory if needed, but here we store all)
# Shape: (simulation_index, time_step, body_index, coordinate_index)
# We only need the final state and the full trajectory for plotting
trajectories = np.zeros((2, NUM_STEPS + 1, num_bodies, 2)) # Sim 1 & Sim 2

# Initialize simulation states
positions_sim1 = np.copy(initial_positions)
velocities_sim1 = np.copy(initial_velocities)
positions_sim2 = np.copy(initial_positions_perturbed)
velocities_sim2 = np.copy(initial_velocities_perturbed)

# Store initial states
trajectories[0, 0] = positions_sim1
trajectories[1, 0] = positions_sim2

start_time = time.time()
for step in range(NUM_STEPS):
    # Update Simulation 1
    positions_sim1, velocities_sim1 = leapfrog_step(positions_sim1, velocities_sim1, masses, DT)
    trajectories[0, step + 1] = positions_sim1

    # Update Simulation 2
    positions_sim2, velocities_sim2 = leapfrog_step(positions_sim2, velocities_sim2, masses, DT)
    trajectories[1, step + 1] = positions_sim2

    # Progress Indicator (print every 1% or so)
    if (step + 1) % (NUM_STEPS // 100) == 0:
        progress = (step + 1) / NUM_STEPS * 100
        elapsed = time.time() - start_time
        print(f"  Progress: {progress:.0f}% complete | Time elapsed: {elapsed:.1f} s", end='\r')

end_time = time.time()
print(f"\n\n--- Simulation Complete ---")
print(f"Total calculation time: {end_time - start_time:.2f} seconds.")
print("-----------------------------------------------------")

# --- VIII. Post-Processing and Visualization ---
print("\nProcessing results for visualization...")

# Convert trajectories from meters to Astronomical Units (AU) for plotting
trajectories_au = trajectories / AU

# Extract final positions
final_pos_sim1 = trajectories_au[0, -1] # Last time step, all bodies, both coordinates
final_pos_sim2 = trajectories_au[1, -1]

# Calculate the final difference for the perturbed planet
final_pos_pert_sim1 = final_pos_sim1[perturb_index]
final_pos_pert_sim2 = final_pos_sim2[perturb_index]
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
              f"Sensitivity to Initial Conditions ({PERTURB_PLANET} perturbed by {PERTURB_FACTOR:.1e})")
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

# Plot trajectories and final positions
for i in range(num_bodies):
    # Simulation 1 (Original) - Solid lines, larger final marker
    ax.plot(trajectories_au[0, :, i, 0], trajectories_au[0, :, i, 1],
            '-', color=colors[i], lw=1, alpha=0.8,
            label=f"{PLANET_NAMES[i]} (Orig)" if i != perturb_index else f"{PLANET_NAMES[i]} (Orig - Solid)")

    ax.plot(final_pos_sim1[i, 0], final_pos_sim1[i, 1],
            'o', color=colors[i], markersize=9 if i==0 else 6, markeredgecolor='white', mew=0.5)

    # Simulation 2 (Perturbed) - Dashed lines, smaller final marker (only for planets)
    if i > 0: # Don't plot perturbed Sun trajectory (it barely moves anyway)
        ax.plot(trajectories_au[1, :, i, 0], trajectories_au[1, :, i, 1],
                '--', color=colors[i], lw=1, alpha=0.7,
                label=f"{PLANET_NAMES[i]} (Perturbed)" if i == perturb_index else None) # Label only perturbed planet

        ax.plot(final_pos_sim2[i, 0], final_pos_sim2[i, 1],
                's', color=colors[i], markersize=4, alpha=0.9) # Use squares for perturbed final pos

# Add an arrow pointing between the final positions of the perturbed planet
ax.annotate('', xy=final_pos_pert_sim2, xytext=final_pos_pert_sim1,
            arrowprops=dict(arrowstyle='<->', color='cyan', lw=1.5))

ax.legend(loc='upper right', fontsize='small', facecolor='dimgray', edgecolor='white', labelcolor='white')
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
plt.show()

print("\n--- End of Script ---")