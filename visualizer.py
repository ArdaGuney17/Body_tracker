import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# --- Configuration ---
# Use the file you uploaded
DATA_FILE = "handover_data_log3.csv"
OUTPUT_FILE = "handover_movement_visualization.html"


def visualize_handover_movement(data_file, output_file):
    """
    Loads landmark data, processes it, and creates an interactive Plotly visualization
    with separate subplots for X and Y coordinate movements over time.
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
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['x_norm'] = pd.to_numeric(df['x_norm'], errors='coerce')
    df['y_norm'] = pd.to_numeric(df['y_norm'], errors='coerce')
    df.dropna(subset=['timestamp', 'x_norm', 'y_norm'], inplace=True)

    # Create a unique identifier for each tracked entity
    df['entity_id'] = df.apply(
        lambda row: f"{row['landmark_name']}_{row.get('handedness', row['source'])}",
        axis=1
    ).str.replace('_HAND_0', '').str.replace('_HAND_1', '')  # Clean up hand IDs for readability

    # 3. Create Subplots (1 row, 2 columns)
    fig = make_subplots(
        rows=1, cols=2,
        shared_xaxes=True,
        subplot_titles=("X Coordinate Movement (Horizontal)", "Y Coordinate Movement (Vertical)")
    )

    # Dictionary to hold legend visibility state to avoid duplicate entries
    show_legend_dict = {}

    for entity in df['entity_id'].unique():
        subset = df[df['entity_id'] == entity]

        # Determine if this is the first time we see this entity ID (for legend display)
        show_in_legend = entity not in show_legend_dict
        show_legend_dict[entity] = True

        # --- Plot X Coordinates (Subplot 1) ---
        fig.add_trace(go.Scatter(
            x=subset['timestamp'],
            y=subset['x_norm'],
            mode='lines',
            name=entity,
            legendgroup=entity,
            showlegend=show_in_legend,
            line=dict(dash='solid')
        ), row=1, col=1)

        # --- Plot Y Coordinates (Subplot 2) ---
        fig.add_trace(go.Scatter(
            x=subset['timestamp'],
            y=subset['y_norm'],
            mode='lines',
            name=entity,
            legendgroup=entity,
            showlegend=False,  # Only show the legend once (in the first subplot)
            line=dict(dash='dot')
        ), row=1, col=2)

    # 4. Configure Layout
    fig.update_layout(
        title_text='Handover Movement Tracking: Separate X and Y Coordinates vs. Time',
        height=700,
        template="plotly_dark",
        hovermode="x unified",
    )

    # Configure X and Y ranges for the subplots
    fig.update_xaxes(title_text='Time (seconds)', row=1, col=1)
    fig.update_xaxes(title_text='Time (seconds)', row=1, col=2)

    fig.update_yaxes(title_text='X Coordinate (Normalized)', range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text='Y Coordinate (Normalized)', range=[0, 1], row=1, col=2)

    # 5. Save and Display
    fig.write_html(output_file)
    print(f"\n--- Visualization saved successfully! ---")
    print(f"Interactive graph saved as: {os.path.abspath(output_file)}")
    print("Open this HTML file in your web browser to view the interactive plot.")


if __name__ == '__main__':
    visualize_handover_movement(DATA_FILE, OUTPUT_FILE)
