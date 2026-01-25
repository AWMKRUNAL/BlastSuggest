import base64
import matplotlib
from flask import Flask, render_template, request
from tabulate import tabulate
matplotlib.rcParams['animation.embed_limit'] = 2**128
app = Flask(__name__)

def calculate_blasting_parameters(user_inputs):
    bench_height = user_inputs["bench_height"]
    rock_density = user_inputs["rock_density"]
    explosive_density = user_inputs["explosive_density"]
    length = user_inputs["length"]
    width = user_inputs["width"]
    hole_diameter = user_inputs["hole_diameter"]
    bedding_condition = user_inputs["bedding_condition"]
    rock_condition = user_inputs["rock_condition"]
    powder_factor = user_inputs["powder_factor"]

    explosive_density_kg_m3 = explosive_density * 1000

    if bedding_condition == "bedding steeply dipping into cut":
        kd = 1.18
    elif bedding_condition == "bedding steeply dipping into face":
        kd = 0.95
    elif bedding_condition == "other cases of deposition":
        kd = 1.00

    if rock_condition == "heavily cracked, frequent weak joints, weakly cemented layers":
        ks = 1.30
    elif rock_condition == "thin well cemented layers with tight joints":
        ks = 1.10
    elif rock_condition == "massive intact rock":
        ks = 0.95
    else:
        raise ValueError("Invalid Rock Condition Value")

    if rock_density == 1.8:
        B1 = 40 * (hole_diameter / 1000)
    elif rock_density == 1.9:
        B1 = 38 * (hole_diameter / 1000)
    elif rock_density == 2.0:
        B1 = 36 * (hole_diameter / 1000)
    elif rock_density == 2.1:
        B1 = 35 * (hole_diameter / 1000)
    elif rock_density == 2.2:
        B1 = 30 * (hole_diameter / 1000)
    elif rock_density == 2.3:
        B1 = 28 * (hole_diameter / 1000)
    elif rock_density == 2.4:
        B1 = 26 * (hole_diameter / 1000)
    elif rock_density == 2.5:
        B1 = 25 * (hole_diameter / 1000)
    elif rock_density >= 2.6:
        B1 = 25 * (hole_diameter / 1000)
    else:
        raise ValueError("Invalid Rock Density Value")

    B2 = 0.012 * (2 * (explosive_density / rock_density) + 1.5) * hole_diameter
    B3 = 8 * 10 ** -3 * hole_diameter * (100 / rock_density) ** (1 / 3)

    B = (B1 + B2 + B3) / 3

    burden = round(kd * ks * B, 2)
    spacing = round(1.35 * burden, 2)

    stemming_distance = 0.7 * burden
    subdrill = 0.3 * burden
    #depth_hole = subdrill + bench_height
    depth_hole = bench_height
    #num_holes_length = int(length / spacing)
    #num_holes_width = int(width / burden)
    #total_holes = num_holes_length * num_holes_width
    x_positions = np.arange(0, length, spacing)
    y_positions = np.arange(0, width, burden)

    # Number of holes calculation
    num_holes_length = len(x_positions)
    num_holes_width = len(y_positions)
    total_holes = num_holes_length * num_holes_width
    total_volume = length * width * bench_height
    total_booster_quantity = round(total_holes * 0.1, 2)
    total_explosive_quantity = round((total_volume / powder_factor) - total_booster_quantity, 2)
    explosive_quantity_per_hole = round(total_explosive_quantity / total_holes, 2)
    charge_height = round(explosive_quantity_per_hole / (explosive_density_kg_m3 * (hole_diameter / 1000) ** 2 * 3.14159 / 4), 2)
    stemming_distance_final = round(depth_hole - charge_height, 2)
    ppv =  round (9907.03 * ((100.0 / (explosive_quantity_per_hole ** 0.5) ** -2.12),2))
    mean_fragmentation_size = round(8 * (burden * spacing * bench_height / explosive_quantity_per_hole) ** 0.8 * explosive_quantity_per_hole ** 0.167,2)
    

    parameters_budgeted_pf = {
        "burden": burden,
        "spacing": spacing,
        "charge_height": charge_height,
        "stemming_distance": stemming_distance_final,
        "hole_depth": depth_hole,
        "bench_height": bench_height,
        "length": length,
        "width": width,
        "total_holes": total_holes,
        "explosive_per_hole": explosive_quantity_per_hole,
        "booster_quantity": total_booster_quantity,
        "total_explosive": total_explosive_quantity,
        "mean_fragmentation_size": mean_fragmentation_size,
        "ppv": ppv,
        "powder_factor": powder_factor
    }
    return parameters_budgeted_pf


# Define or import the suggest_improvements function as needed
def suggest_improvements(user_inputs):
    bench_height = user_inputs["bench_height"]
    rock_density = user_inputs["rock_density"]
    explosive_density = user_inputs["explosive_density"]
    length = user_inputs["length"]
    width = user_inputs["width"]
    hole_diameter = user_inputs["hole_diameter"]
    bedding_condition = user_inputs["bedding_condition"]
    rock_condition = user_inputs["rock_condition"]
    powder_factor = user_inputs["powder_factor"]

    suggested_burden_sr3 = bench_height / 3
    suggested_burden_sr4 = bench_height / 4

    #if initiation_type == "instantaneous":
        #if bench_height / suggested_burden_sr3 < 4:
            #spacing_sr3 = (bench_height + 2 * suggested_burden_sr3) / 3
        #else:
            #spacing_sr3 = 2 * suggested_burden_sr3
        #if bench_height / suggested_burden_sr4 < 4:
            #spacing_sr4 = (bench_height + 2 * suggested_burden_sr4) / 3
        #else:
            #spacing_sr4 = 2 * suggested_burden_sr4

    #elif initiation_type == "delayed":
        #if bench_height / suggested_burden_sr3 < 4:
            #spacing_sr3 = (bench_height + 7 * suggested_burden_sr3) / 8
        #else:
            #spacing_sr3 = 1.4 * suggested_burden_sr3
        #if bench_height / suggested_burden_sr4 < 4:
            #spacing_sr4 = (bench_height + 7 * suggested_burden_sr4) / 8
        #else:
            #spacing_sr4 = 1.4 * suggested_burden_sr4

    Burden1 = (suggested_burden_sr3 + suggested_burden_sr4) / 2


    if bedding_condition == "bedding steeply dipping into cut":
        kd1 = 1.18
    elif bedding_condition == "bedding steeply dipping into face":
        kd1 = 0.95
    elif bedding_condition == "other cases of deposition":
        kd1 = 1.00

    if rock_condition == "heavily cracked, frequent weak joints, weakly cemented layers":
        ks1 = 1.30
    elif rock_condition == "thin well cemented layers with tight joints":
        ks1 = 1.10
    elif rock_condition == "massive intact rock":
        ks1 = 0.95
    else:
        raise ValueError("Invalid Rock Condition Value")

    if rock_density == 1.8:
        Burden2 = 40 * (hole_diameter / 1000)
    elif rock_density == 1.9:
        Burden2 = 38 * (hole_diameter / 1000)
    elif rock_density == 2.0:
        Burden2 = 36 * (hole_diameter / 1000)
    elif rock_density == 2.1:
        Burden2 = 35 * (hole_diameter / 1000)
    elif rock_density == 2.2:
        Burden2 = 30 * (hole_diameter / 1000)
    elif rock_density == 2.3:
        Burden2 = 28 * (hole_diameter / 1000)
    elif rock_density == 2.4:
        Burden2 = 26 * (hole_diameter / 1000)
    elif rock_density == 2.5:
        Burden2 = 25 * (hole_diameter / 1000)
    elif rock_density >= 2.6:
        Burden2 = 25 * (hole_diameter / 1000)
    else:
        raise ValueError("Invalid Rock Density Value")

    Burden3 = 0.012 * (2 * (explosive_density / rock_density) + 1.5) * hole_diameter
    Burden4 = 8 * 10 ** -3 * hole_diameter * (100 / rock_density) ** (1 / 3)
    import statistics
    Burden5 = statistics.median([Burden1,Burden2,Burden3,Burden4])
    average_burden = kd1 * ks1 * Burden5

    #if rock_density >= 2.2:
        #spacing2 = 1.2 * average_burden
    #elif rock_density < 2.2:
        #spacing2 = 1.5 * average_burden

    #average_spacing = (Spacing1 + spacing2) / 2
    average_spacing = 1.35 * average_burden
    subdrill = 0.3 * average_burden
    hole_depth = bench_height
    #hole_depth = subdrill + bench_height
    #num_holes_length = int(length / average_spacing)
    #num_holes_width = int(width / average_burden)
    x_positions = np.arange(0, length, average_spacing)
    y_positions = np.arange(0, width,average_burden )

    # Number of holes calculation
    num_holes_length = len(x_positions)
    num_holes_width = len(y_positions)
    #total_holes = num_holes_length * num_holes_width
    total_holes = num_holes_length * num_holes_width
    total_volume = length * width * bench_height
    total_booster_quantity = total_holes * 0.1
    total_explosive_quantity = ( total_volume / powder_factor) - total_booster_quantity
    explosive_per_hole = total_explosive_quantity / total_holes
    charge_height = explosive_per_hole / ((explosive_density * 1000) * (hole_diameter / 1000) ** 2 * 3.14159 / 4)
    stemming_distance = hole_depth - charge_height
    ppv = 9907.03 * ((100.0 / (total_explosive_quantity / total_holes) ** 0.5) ** -2.12)
    mean_fragmentation_size = 8 * (average_burden * average_spacing * bench_height / explosive_per_hole) ** 0.8 * explosive_per_hole ** 0.167
    

    range_factor = 0.15
    ranges = {
        "burden": (average_burden * (1 - range_factor), average_burden * (1 + range_factor)),
        "spacing": (average_spacing * (1 - range_factor), average_spacing * (1 + range_factor)),
        "charge_height": (charge_height * (1 - range_factor), charge_height * (1 + range_factor)),
        "stemming_distance": (stemming_distance * (1 - range_factor), stemming_distance * (1 + range_factor)),
        "hole_depth": (hole_depth * (1 - range_factor), hole_depth * (1 + range_factor)),
        "total_holes": (total_holes * (1 - range_factor), total_holes * (1 + range_factor)),
        "bench_height": (bench_height * (1 - range_factor), bench_height * (1 + range_factor)),
        "length":length,
        "width": width,
        "explosive_per_hole": (explosive_per_hole * (1 - range_factor), explosive_per_hole * (1 + range_factor)),
        "booster_quantity": (total_booster_quantity * (1 - range_factor), total_booster_quantity * (1 + range_factor)),
        "total_explosives": (total_explosive_quantity * (1 - range_factor), total_explosive_quantity * (1 + range_factor)),
        "mean_fragmentation_size": (mean_fragmentation_size * (1 - range_factor), mean_fragmentation_size * (1 + range_factor)),
        "ppv": (ppv * (1 - range_factor), ppv * (1 + range_factor)),
        "powder_factor": (powder_factor * (1 - range_factor), powder_factor * (1 + range_factor))
    }

    return {
        "burden": average_burden,
        "spacing": average_spacing,
        "subdrill": subdrill,
        "hole_depth": hole_depth,
        "stemming_distance": stemming_distance,
        "charge_height": charge_height,
        "bench_height": bench_height,
        "length": length,
        "width": width,
        "explosive_per_hole": explosive_per_hole,
        "total_volume": total_volume,
        "total_holes": total_holes,
        "total_explosives": total_explosive_quantity,
        "booster_quantity": total_booster_quantity,
        "mean_fragmentation_size": mean_fragmentation_size,
        "ppv":ppv,
        "powder_factor": powder_factor,
        "ranges": ranges
    }


def generate_blasting_pattern(user_inputs, spacing, burden, total_holes):
    """
    Optimize the blasting pattern generation by limiting grid size.
    """
    rock_density = user_inputs["rock_density"]
    width = user_inputs["width"]
    length = user_inputs["length"]

    max_holes = 500  # Define a max limit for total holes

    # Adjust grid positions if holes exceed max limit
    rows = int(length / spacing)
    cols = int(width / burden)
    if total_holes > max_holes:
        rows = rows // 2
        cols = cols // 2

    positions = []
    for row in range(rows):
        for col in range(cols):
            if rock_density < 2.2:  # Square pattern
                positions.append((col * spacing, row * burden))
            else:  # Staggered pattern
                x_offset = col * spacing + (spacing / 2 if row % 2 == 1 else 0)
                positions.append((x_offset, row * burden))

    return positions, rows



from matplotlib.animation import FuncAnimation


import os
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def plot_blasting_pattern(positions, burden, spacing, num_rows, connection_type, row_delay=None, diagonal_delay=None,
                          pattern_type=None, image_path=None, img_byte_stream=None):
    x, y = zip(*positions)
    fig, ax = plt.subplots(figsize=(12, 6))

    # Check if the image byte stream is provided
    if img_byte_stream is not None:
        try:
            img = mpimg.imread(io.BytesIO(img_byte_stream), format='png')  # Set the correct format if necessary
            # Plot background image
            ax.imshow(img)
        except Exception as e:
            print(f"Error loading image from byte stream: {e}")
    elif image_path and os.path.exists(image_path):
        # Load the background image from the file
        img = mpimg.imread(image_path)
        # Plot background image
        ax.imshow(img)
    else:
        print(f"Warning: No valid image source provided. Skipping background image.")

    scatter = ax.scatter(x, y, c='blue', s=100, edgecolors='black')

    delays = [None] * len(positions)
    last_row_start = (num_rows - 1) * (len(positions) // num_rows)
    delays[last_row_start] = 0

    if row_delay is not None:
        for i in range(last_row_start + 1, len(positions)):
            delays[i] = delays[i - 1] + row_delay if delays[i - 1] is not None else row_delay

        for row in range(num_rows - 2, -1, -1):
            row_start = row * (len(positions) // num_rows)
            for i in range(row_start, row_start + (len(positions) // num_rows)):
                if i % (len(positions) // num_rows) == 0:
                    if row % 2 == 1:
                        delays[i] = delays[i + (len(positions) // num_rows) + 1] + (
                            diagonal_delay if diagonal_delay is not None else 0)
                    else:
                        delays[i] = delays[i + (len(positions) // num_rows)] + (
                            diagonal_delay if diagonal_delay is not None else 0)
                else:
                    delays[i] = delays[i - 1] + row_delay if delays[i - 1] is not None else row_delay

    ax.grid(False)
    ax.set_xlim(-spacing, max(x) + spacing)
    ax.set_ylim(-spacing, max(y) + burden)
    ax.set_aspect('equal', adjustable='box')

    arrows = []

    def add_arrow(start_x, start_y, end_x, end_y, color):
        arrow = ax.arrow(start_x, start_y, end_x - start_x, end_y - start_y, head_width=0.1, head_length=0.1, fc=color,
                         ec=color)
        arrows.append(arrow)

    if connection_type == 'diagonal':
        for i in range(len(positions) - 1):
            if y[i] == y[i + 1] and y[i] == (num_rows - 1) * burden:
                add_arrow(x[i], y[i], x[i + 1], y[i], 'black')

        for row in range(num_rows - 1, 0, -1):
            for i in range(1, len(positions)):
                if y[i] == row * burden:
                    current_x = x[i]
                    current_y = y[i]
                    while True:
                        next_x = current_x - spacing / 2
                        next_y = current_y - burden
                        if (next_x, next_y) in positions:
                            add_arrow(current_x, current_y, next_x, next_y, 'red')
                            current_x = next_x
                            current_y = next_y
                        else:
                            break

        for i in range(len(positions) - 1):
            if y[i] == (num_rows - 1) * burden:
                for j in range(len(positions)):
                    if x[j] == x[i] + spacing and y[j] == y[i]:
                        add_arrow(x[i], y[i], x[j], y[j], 'black')

        for i in range(len(positions) - 1):
            if y[i] % (2 * burden) != 0:
                if x[i] == max(x) - spacing:
                    add_arrow(x[i], y[i], x[i] + spacing, y[i], 'black')

        for row in range(num_rows - 2, 0, -1):
            for i in range(len(positions) - 1):
                if y[i] == row * burden and x[i] == max(x):
                    current_x = x[i]
                    current_y = y[i]
                    while True:
                        next_x = current_x - spacing / 2
                        next_y = current_y - burden
                        if (next_x, next_y) in positions:
                            add_arrow(current_x, current_y, next_x, next_y, 'red')
                            current_x = next_x
                            current_y = next_y
                        else:
                            break

        for i in range(len(positions) - 1):
            if y[i] == (num_rows - 2) * burden and x[i] == max(x):
                for j in range(len(positions)):
                    if y[j] == (num_rows - 2) * burden and x[j] == x[i] - spacing:
                        add_arrow(x[j], y[j], x[i], y[j], 'black')

        for i in range(len(positions)):
            if y[i] == min(y):
                for j in range(len(positions)):
                    if y[j] == min(y) and x[j] == max(x):
                        add_arrow(x[j - 1], y[j - 1], x[j], y[j], 'black')

    elif connection_type == 'line':
        for row in range(num_rows):
            row_positions = [pos for pos in positions if pos[1] == row * burden]
            for i in range(len(row_positions) - 1):
                add_arrow(row_positions[i][0], row_positions[i][1], row_positions[i + 1][0], row_positions[i + 1][1],
                          'black')

        for row in range(num_rows - 1, 0, -1):
            for i in range(1, len(positions)):
                if y[i] == row * burden:
                    current_x = x[i]
                    current_y = y[i]
                    next_x = current_x - spacing / 2
                    next_y = current_y - burden
                    if (next_x, next_y) in positions:
                        add_arrow(current_x, current_y, next_x, next_y, 'red')
                        break

    return fig, ax, scatter, delays


import plotly.graph_objects as go
import numpy as np


def create_animation_plotly(positions, delays, spacing, burden):
    """
    Optimized Plotly animation function to reduce frames for large datasets.
    """

    x, y = zip(*positions)
    max_x = max(x) + spacing
    max_y = max(y) + burden
    unique_delays = sorted(set(delays))  # Only include significant frames
    delay_frames = max(delays) + 10

    # Create frames for significant changes only
    frames = []
    for frame in unique_delays:  # Use unique delays to reduce frames
        frame_data = go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=[40 if frame >= delay else 20 for delay in delays],
                color=['red' if frame >= delay else 'blue' for delay in delays],
                symbol=['circle' if frame >= delay else 'circle' for delay in delays]
            ),
        )
        frames.append(go.Frame(data=[frame_data], name=f"frame_{frame}"))

    scatter_init = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        marker=dict(size=[20] * len(positions), color="blue", symbol="square")
    )

    fig = go.Figure(
        data=[scatter_init],
        layout=go.Layout(
            xaxis=dict(range=[-spacing / 2, max_x]),
            yaxis=dict(range=[-burden / 2, max_y]),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(label="Play", method="animate", args=[None, {"fromcurrent": True}]),
                        dict(label="Pause", method="animate",
                             args=[[None], {"frame": {"duration": 0, "redraw": False}}])
                    ],
                )
            ],
        ),
        frames=frames,
    )

    fig.update_layout(
        autosize=True,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig



def draw_combined_pattern(length, width, spacing, burden, bench_height, pattern_type, hole_details):
    """
    Reduce hole density for large grids to optimize performance.
    """
    max_points = 500  # Limit the number of grid points
    x_positions = np.arange(0, length, spacing)
    y_positions = np.arange(0, width, burden)

    if len(x_positions) * len(y_positions) > max_points:
        x_positions = x_positions[::2]  # Reduce density by skipping points
        y_positions = y_positions[::2]

    fig = go.Figure()

    # Add combined holes and the surface pattern
    add_holes(fig, length, width, spacing, burden, hole_details["charge_height"],
              hole_details["stemming_distance"], hole_details["hole_depth"], pattern_type)

    x_grid, y_grid = np.meshgrid(x_positions, y_positions)
    z_grid = np.full_like(x_grid, hole_details["hole_depth"], dtype=float)

    fig.add_trace(
        go.Surface(
            x=x_grid,
            y=y_grid,
            z=z_grid,
            colorscale="Greys",
            opacity=0.5,
            showscale=False
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Length (m)'),
            yaxis=dict(title='Width (m)'),
            zaxis=dict(title='Depth (m)', range=[0, hole_details["hole_depth"] + 0.5]),
        )
    )

    return fig.to_html(full_html=False)



def add_holes(fig, length, width, spacing, burden, charge_height, stemming_distance, depth_hole, pattern_type):
    x_positions = np.arange(0, length, spacing)
    y_positions = np.arange(0, width, burden)

    for i, y in enumerate(y_positions):
        if pattern_type == "staggered" and i % 2 == 1:
            # Add the staggered pattern, ensuring it doesn't introduce an extra hole
            x_positions_row = x_positions + spacing / 2
            #x_positions_row = x_positions_row[x_positions_row < length]  # Filter out the extra hole position
        else:
            x_positions_row = x_positions

        for x in x_positions_row:
            #if x < length:  # Ensure the hole position is within the length of the grid
            fig.add_trace(create_hole(x, y, charge_height, stemming_distance, depth_hole))



def create_hole(x, y, charge_height, stemming_distance, depth_hole):
    # Setting marker size to some positive value for the top, and zero for the others
    marker_sizes = [0, 0, 0, 10]
    # Setting marker color as a valid color for the top, and fully transparent for the others
    marker_colors = ['rgba(0,0,0,0)', 'rgba(0,0,0,0)', 'rgba(0,0,0,0)','black',]

    hole = go.Scatter3d(
        x=[x, x, x, x],
        y=[y, y, y, y],
        z=[0, charge_height, stemming_distance, depth_hole],
        mode="lines+markers",
        line=dict(color=['black', 'grey', 'orange'], width=10),
        marker=dict(color=marker_colors, size=marker_sizes)
    )
    return hole


def draw_single_diagram(hole_details):
    explosive_density_kg_m3 = hole_details["explosive_density_g_cm3"] * 1000
    charge_height = hole_details["explosive_quantity_kg"] / (
            explosive_density_kg_m3 * (hole_details["diameter_mm"] / 1000) ** 2 * 3.141592653589793 / 4)
    stemming_distance_m = hole_details["depth_m"] - charge_height
    fig, ax = plt.subplots()

    # Draw the explosive charge
    charge = plt.Rectangle((0.5 - hole_details["diameter_mm"] / 2000, hole_details["depth_m"] - charge_height),
                           hole_details["diameter_mm"] / 1000, charge_height, edgecolor='black', facecolor='black',
                           label='Explosive Charge')
    ax.add_patch(charge)

    # Draw the stemming
    stemming = plt.Rectangle((0.5 - hole_details["diameter_mm"] / 2000, 0), hole_details["diameter_mm"] / 1000,
                             stemming_distance_m, edgecolor='black', facecolor='grey', label='Stemming Distance')
    ax.add_patch(stemming)

    # Draw the void space
    void_space_height = hole_details["depth_m"] - charge_height - stemming_distance_m
    void_space = plt.Rectangle((0.5 - hole_details["diameter_mm"] / 2000, stemming_distance_m),
                               hole_details["diameter_mm"] / 1000, void_space_height, edgecolor='black',
                               facecolor='none', label='Void Space')
    ax.add_patch(void_space)

    # Draw the nonel line
    nonel_line_length = hole_details["depth_m"] + 1
    nonel_line = plt.Line2D([0.5] * 2, [hole_details["depth_m"] - nonel_line_length, hole_details["depth_m"] - 0.2],
                            color='orange', linewidth=2, label='Nonel Line')
    ax.add_line(nonel_line)

    # Draw the booster
    booster_square = plt.Rectangle((0.5 - hole_details["diameter_mm"] / 2000 / 2, hole_details['depth_m'] - 0.2),
                                   hole_details['diameter_mm'] / 1000, 0.2, edgecolor='black', facecolor='yellow',
                                   label='yellow')
    ax.add_patch(booster_square)

    # Annotations
    ax.annotate(f'Depth: {hole_details["depth_m"]} m',
                xy=(0.5 + hole_details["diameter_mm"] / 2000 / 2, hole_details["depth_m"]),
                xytext=(1.5, hole_details["depth_m"]), arrowprops=dict(facecolor='black', shrink=0.05, width=1),
                ha='center')

    ax.annotate(f'Charge Height: {charge_height:.2f} m',
                xy=(0.5 + hole_details["diameter_mm"] / 2000 / 2, hole_details["depth_m"] - charge_height / 2),
                xytext=(1.5, hole_details["depth_m"] - charge_height / 2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1),
                ha='center', color='black')

    ax.annotate(f'Stemming Distance: {stemming_distance_m:.2f} m',
                xy=(0.5 + hole_details["diameter_mm"] / 2000 / 2, stemming_distance_m / 2),
                xytext=(1.5, stemming_distance_m / 2), arrowprops=dict(facecolor='green', shrink=0.05, width=1),
                ha='center', color='grey')

    ax.set_xlim(0, 2)
    ax.set_ylim(hole_details["depth_m"] + 1, -1)
    ax.set_xlabel('Horizontal Position (scaled)')
    ax.set_ylabel('Depth (m)')
    ax.legend()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)
    return base64.b64encode(img.getvalue()).decode('utf-8')


@app.route('/')
def index():
    return render_template('index.html')




@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        try:
            # Convert and collect user inputs
            user_input = {
                "bench_height": float(request.form["bench_height"]),
                "rock_density": float(request.form["rock_density"]),
                "explosive_density": float(request.form["explosive_density"]),
                "length": float(request.form["length"]),
                "width": float(request.form["width"]),
                "hole_diameter": float(request.form["hole_diameter"]),
                "bedding_condition": request.form["bedding_condition"],
                "rock_condition": request.form["rock_condition"],
                "powder_factor": float(request.form["powder_factor"]),
            }

            # Input validation
            if user_input["length"] > 1000 or user_input["width"] > 1000:
                return "<h1>Error: Length and Width values are too large. Please reduce the dimensions.</h1>"

            if user_input["hole_diameter"] < 1 or user_input["hole_diameter"] > 1000:
                return "<h1>Error: Invalid Hole Diameter. Please provide a valid range (1-1000 mm).</h1>"

            # Parameter names mapping
            parameter_names = {
                "bench_height": "Bench Height (m)",
                "rock_density": "Rock Density (g/cm³)",
                "explosive_density": "Explosive Density (g/cm³)",
                "length": "Length (m)",
                "width": "Width (m)",
                "hole_diameter": "Hole Diameter (mm)",
                "bedding_condition": "Bedding Condition",
                "rock_condition": "Rock Condition",
                "powder_factor": "Powder Factor",
                "burden": "Burden (m)",
                "spacing": "Spacing (m)",
                "charge_height": "Charge Height (m)",
                "stemming_distance": "Stemming Distance (m)",
                "hole_depth": "Hole Depth (m)",
                "total_holes": "Total Holes",
                "explosive_per_hole": "Explosive per Hole (kg)",
                "booster_quantity": "Booster Quantity (kg)",
                "total_explosive": "Total Explosive (kg)",
                "mean_fragmentation_size": "Mean Fragmentation Size (cm)"
            }

            # Calculate blasting parameters and suggest improvements
            params_budgeted = calculate_blasting_parameters(user_input)
            suggestions = suggest_improvements(user_input)
            parameters = list(params_budgeted.keys())

            # Prepare data for the table
            table_data = []
            user_input_parameters = {
                "bench_height": user_input["bench_height"],
                "rock_density": user_input["rock_density"],
                "explosive_density": user_input["explosive_density"],
                "length": user_input["length"],
                "width": user_input["width"],
                "hole_diameter": user_input["hole_diameter"],
                "bedding_condition": user_input["bedding_condition"],
                "rock_condition": user_input["rock_condition"],
                "powder_factor": user_input["powder_factor"]
            }

            for param, value in user_input_parameters.items():
                table_data.append([parameter_names[param], value, value])

            # Update the table for permissible ranges
            for param in parameters:
                budgeted_value = params_budgeted.get(param, "N/A")

                range_key = "total_explosives" if param == "total_explosive" else param

                if "ranges" in suggestions and range_key in suggestions["ranges"]:
                    if isinstance(suggestions["ranges"][range_key], (tuple, list)) and len(
                            suggestions["ranges"][range_key]) == 2:
                        range_min, range_max = suggestions["ranges"][range_key]
                        table_data.append(
                            [parameter_names[param], budgeted_value, f"{range_min:.2f} - {range_max:.2f}"]
                        )
                    else:
                        table_data.append([parameter_names[param], budgeted_value, budgeted_value])
                else:
                    table_data.append([parameter_names[param], budgeted_value, budgeted_value])

            # Generate the HTML table
            headers = ["Parameter", "Value", "Permissible Range"]
            table_html = tabulate(table_data, headers, tablefmt="html")

            # Drawing combined pattern and single diagram
            combined_pattern_img_html = draw_combined_pattern(
                length=user_input["length"],
                width=user_input["width"],
                spacing=params_budgeted["spacing"],
                burden=params_budgeted["burden"],
                bench_height=params_budgeted["bench_height"],
                pattern_type='staggered' if user_input["rock_density"] >= 2.2 else 'square',
                hole_details=params_budgeted
            )

            hole_details = {
                "diameter_mm": user_input["hole_diameter"],
                "depth_m": params_budgeted["hole_depth"],
                "explosive_density_g_cm3": user_input["explosive_density"],
                "explosive_quantity_kg": params_budgeted["explosive_per_hole"],
            }
            single_diagram_img = draw_single_diagram(hole_details)

            # Determine pattern type based on rock density
            pattern_type = 'square' if user_input["rock_density"] < 2.2 else 'staggered'

            # Generate positions based on calculated parameters
            positions, num_rows = generate_blasting_pattern(user_input, params_budgeted["spacing"],
                                                            params_budgeted["burden"],
                                                            params_budgeted["total_holes"])

            # Assume image_path is the path to your image file
            image_path = 'path/to/uploaded/image.jpg'

            # Read the image file into a byte stream for flexibility
            img_byte_stream = None
            if os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    img_byte_stream = f.read()

            # Plot the blasting pattern
            fig, ax, scatter, delays = plot_blasting_pattern(
                positions,
                params_budgeted["burden"],
                params_budgeted["spacing"],
                num_rows,
                connection_type='diagonal',
                row_delay=17,
                diagonal_delay=42,
                pattern_type='example_pattern_type',  # Replace with actual pattern type
                image_path=image_path,  # Provide the image path
                img_byte_stream=img_byte_stream  # Provide the image byte stream
            )

            # Create animation
            # Generate animation using Plotly
            animation_html = create_animation_plotly(
                positions=positions,
                delays=delays,
                spacing=params_budgeted["spacing"],
                burden=params_budgeted["burden"]
            ).to_html(full_html=False)

            return render_template('result.html', table_html=table_html,
                                   combined_pattern_img_html=combined_pattern_img_html,
                                   single_diagram_img=single_diagram_img,
                                   animation_html=animation_html)

        except (ValueError, KeyError) as e:
            return f"<h1>Error: {e}</h1>"
        except Exception as e:
            return f"<h1>Unexpected Error: {e}</h1>"


if __name__ == "__main__":
    app.run(debug=True)




