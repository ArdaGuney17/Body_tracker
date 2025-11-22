import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Configuration ---
DATA_FILE = "handover_data_log0.csv"
OUTPUT_FILE = "handover_trajectory_plot_full.png"  # Output file name changed for clarity

# Sample rate: Only plot every Nth frame to keep the visualization clean
# (Adjust this value based on the total length of your recording)
SAMPLE_RATE = 1  # CHANGED: Set to 1 to plot every single captured point!


def visualize_handover_trajectory(data_file, output_file, sample_rate):
    """
    Loads landmark data and plots the X vs Y position (trajectory) over time
    using Matplotlib to generate a static image plot.
    """
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at '{data_file}'. Please ensure the file is in the same directory.")
        return

    # 1. Load Data
    try:
        df = pd.read_csv(data_file)
        print(f"Successfully loaded {len(df)} records from {data_file}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # 2. Data Cleaning and Preparation
    df['frame_id'] = pd.to_numeric(df['frame_id'], errors='coerce')
    df['x_norm'] = pd.to_numeric(df['x_norm'], errors='coerce')
    df['y_norm'] = pd.to_numeric(df['y_norm'], errors='coerce')
    df.dropna(subset=['frame_id', 'x_norm', 'y_norm'], inplace=True)

    # Create a unique identifier for each tracked entity
    df['entity_id'] = df.apply(
        lambda row: f"{row['landmark_name']}_{row.get('handedness', row['source'])}",
        axis=1
    ).str.replace('_HAND_0', '').str.replace('_HAND_1', '')  # Clean up hand IDs for readability

    # 3. Downsample the data for cleaner trajectory visualization
    # We only keep rows where the frame_id is divisible by the sample rate
    # If SAMPLE_RATE is 1, this keeps every point.
    df_sampled = df[df['frame_id'] % sample_rate == 0].copy()

    if sample_rate == 1:
        print(f"Plotting all {len(df_sampled)} captured points.")
    else:
        print(f"Downsampled from {len(df)} to {len(df_sampled)} records (Rate: 1/{sample_rate})")

    # 4. Create Trajectory Plot (Matplotlib)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Loop through each unique tracked entity
    for entity in df_sampled['entity_id'].unique():
        subset = df_sampled[df_sampled['entity_id'] == entity]

        # Plot the trajectory line. Set markersize small and use linewidth to trace the path.
        line, = ax.plot(
            subset['x_norm'],
            subset['y_norm'],
            label=entity,
            marker='.',  # CHANGED: Use a small dot marker for high density
            markersize=2,  # CHANGED: Small marker size
            linestyle='-',
            linewidth=0.5  # CHANGED: Thin line to connect the dots
        )

        # Mark Start and End points for clear time progression
        color = line.get_color()

        # Mark Start (bigger marker: triangle up)
        start_point = subset.iloc[0]
        ax.plot(start_point['x_norm'], start_point['y_norm'], marker='^', markersize=8, color=color,
                label=f'{entity} Start', linestyle='', alpha=0.9, zorder=5)  # zorder to ensure it's on top

        # Mark End (bigger marker: square)
        end_point = subset.iloc[-1]
        ax.plot(end_point['x_norm'], end_point['y_norm'], marker='s', markersize=8, color=color,
                label=f'{entity} End', linestyle='', alpha=0.9, zorder=5)  # zorder to ensure it's on top

    # 5. Configure Axes and Layout

    # Invert Y-axis so (0,0) is top-left, matching camera view
    ax.invert_yaxis()

    # Set aspect ratio to equal to represent actual space accurately (normalized 0-1)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # Set Y limits after inversion

    ax.set_title('Full Trajectory of Handover Landmarks (X vs. Y) - All Points', fontsize=14)
    ax.set_xlabel('Normalized X Position (Horizontal)', fontsize=12)
    ax.set_ylabel('Normalized Y Position (Vertical)', fontsize=12)

    # Place legend outside the plot area for clarity, combining initial and end points
    # We filter out the 'Start' and 'End' legends to keep only the main trajectory name
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {}
    for handle, label in zip(handles, labels):
        # Keep only the main entity label
        if not label.endswith('Start') and not label.endswith('End'):
            unique_labels[label] = handle

    ax.legend(unique_labels.values(), unique_labels.keys(), title="Tracked Landmark", bbox_to_anchor=(1.05, 1),
              loc='upper left')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout for outside legend

    # 6. Save the figure as a PNG image
    plt.savefig(output_file)
    print(f"\n--- Visualization saved successfully! ---")
    print(f"Static plot saved as: {os.path.abspath(output_file)}")
    print("This is a static PNG image and will not be interactive.")
    plt.close(fig)  # Close figure to free up memory


if __name__ == '__main__':
    # You must install matplotlib: pip install matplotlib
    visualize_handover_trajectory(DATA_FILE, OUTPUT_FILE, SAMPLE_RATE)
