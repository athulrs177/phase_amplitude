import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def draw_filled_arc(radius, theta1, theta2, col, zorder, alpha):
    """
    Draw a filled arc on a plot. This is required for the next function.

    Parameters:
    - radius (float): Radius of the arc.
    - theta1 (float): Starting angle of the arc in degrees.
    - theta2 (float): Ending angle of the arc in degrees.
    - col (str): Color of the filled arc.
    - zorder (int): Z-order of the filled arc (determines drawing order).
    - alpha (float): Alpha (transparency) value for the filled arc.

    Returns:
    None

    Example:
    >>> draw_filled_arc(radius=5, theta1=45, theta2=135, col='blue', zorder=2, alpha=0.5)
    """
    # Generate points along the arc
    theta = np.linspace(np.radians(theta1), np.radians(theta2), 100)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Plot lines connecting edges of the arc to the center
    center_x = 0  # X-coordinate of the center
    center_y = 0  # Y-coordinate of the center
    start_x = radius * np.cos(np.radians(theta1))
    start_y = radius * np.sin(np.radians(theta1))
    end_x = radius * np.cos(np.radians(theta2))
    end_y = radius * np.sin(np.radians(theta2))

    # Generate points for the filled region
    theta_fill = np.linspace(np.radians(theta1), np.radians(theta2), 100)
    x_fill = np.concatenate(([center_x], radius * np.cos(theta_fill), [center_x]))
    y_fill = np.concatenate(([center_y], radius * np.sin(theta_fill), [center_y]))

    # Plot the filled region
    plt.fill_between(x_fill, y_fill, color=col, alpha=alpha, edgecolor=None, zorder=zorder)

def plot_normalized_scatter(A, stdA, std_dA, quantile1, quantile2, ax, wave, time, whiten_center=True, whiten_threshold=1.0):
    """
    Plot a scatter plot of normalized values and highlight the phases.

    Parameters:
    - A (xarray.DataArray): Input data to be plotted.
    - stdA (float): standard deviation of over all times and longitudes of A
    - std_dA (float): standard deviation of over all times and longitudes of dA/dt
    - quantile1 (float): The quantile value (in percentage) for the lower percentile region (e.g., 95th percentile).
    - quantile2 (float): The quantile value (in percentage) for the higher percentile region (e.g., 99th percentile).
    - ax (matplotlib.axes.Axes): Axes object for the plot.
    - wave (str): Type of wave (e.g., 'MJO' or 'ER').
    - time (str): Time variable used for selecting specific time points to compute climatology.
    - whiten_center (boolean): Whether the central region (in this version, less than 'whiten_threshold' std) needs to be masked
      default_value: True
    - whiten_threshold: amplitude of the central region that gets masked if whiten_center=True

    """

    # Normalize A by subtracting the mean and dividing by the standard deviation
    normalized_A = (A - np.nanmean(A)) / stdA

    # Calculate the first derivative of A
    first_derivative = A.differentiate('time')#np.gradient(A)

    # Normalize the first derivative by subtracting the mean and dividing by the standard deviation
    normalized_derivative = (first_derivative - np.nanmean(first_derivative)) / std_dA

    normalized_A = normalized_A.sel(time=time)
    normalized_derivative = normalized_derivative.sel(time=time)

    # Compute distances
    distances = np.sqrt(normalized_A**2 + normalized_derivative**2)

    # Set the plot limits to center it around (0, 0)
    max_abs_value = max(np.abs(normalized_A.min()), np.abs(normalized_A.max()),
                        np.abs(normalized_derivative.min()), np.abs(normalized_derivative.max()))

    # Classify the points into phases based on the given rule
    theta = np.arctan2(normalized_derivative, normalized_A) * (180 / np.pi)
    theta[theta < 0] += 360
    phase = np.zeros_like(theta, dtype=int)
    phase[(theta >= 22.5) & (theta < 67.5)] = 4
    phase[(theta >= 67.5) & (theta < 112.5)] = 3
    phase[(theta >= 112.5) & (theta < 157.5)] = 2
    phase[(theta >= 157.5) & (theta < 202.5)] = 1
    phase[(theta >= 202.5) & (theta < 247.5)] = 8
    phase[(theta >= 247.5) & (theta < 292.5)] = 7
    phase[(theta >= 292.5) & (theta < 337.5)] = 6
    phase[(theta >= 337.5) | (theta < 22.5)] = 5

    angles_phase = [(157.5, 202.5), (112.5, 157.5), (67.5, 112.5), (22.5, 67.5),
                    (-22.5, 22.5), (292.5, 337.5), (247.5, 292.5), (202.5, 247.5)
                    ]

    # Iterate over each phase and plot shaded arced regions for the 95th and 99th percentile distances
    c95 = 'C1'
    c99 = 'C0'
    for p in range(1, 9):
        phase_points = distances[phase == p]
        if len(phase_points) > 0:
            percentile_95 = np.percentile(phase_points, quantile1)
            percentile_99 = np.percentile(phase_points, quantile2)

            # Draw the 99th percentile region
            draw_filled_arc(percentile_99, angles_phase[p-1][0], angles_phase[p-1][1], c99, zorder=0, alpha=1.0)

            # Draw the 95th percentile region
            draw_filled_arc(percentile_95, angles_phase[p-1][0], angles_phase[p-1][1], c95, zorder=1, alpha=1.0)

    # Plot lines to divide the phases based on the phase angles
    for angle in np.arange(22.5, 382.5, 45):
        rad_angle = np.deg2rad(angle)
        x = 25 * np.cos(rad_angle)
        y = 25 * np.sin(rad_angle)
        ax.plot([-x, x], [-y, y], color='red', linestyle='--', lw=0.5, zorder=3)

    # Uncomment if full scatter is needed
    ax.scatter(normalized_A, normalized_derivative, color='black', s=0.25, zorder=4)

    # Whiten the 1 std region
    if whiten_center:
        draw_filled_arc(whiten_threshold, 0, 360, 'white', zorder=5, alpha=1.0)

    # Show the plot
    ax.set_xlim(-max_abs_value - 0.5, max_abs_value + 0.5)
    ax.set_ylim(-max_abs_value - 0.5, max_abs_value + 0.5)

    custom_legend = [
        mpatches.Patch(color=c95),
        mpatches.Patch(color=c99),
        mpatches.Patch(color='white')
    ]
    legend_labels = [
        '<' + str(quantile1) + 'th percentile',
        str(quantile1) + 'th-' + str(quantile2) + 'th percentile',
        '>' + str(quantile2) + 'th percentile'
    ]

    ax.legend(custom_legend, legend_labels, loc='upper right', fontsize=8)

    # Change this according to the dataset
    ax.set_xlabel('normalized IMERG')
    ax.set_ylabel('normalized d(IMERG)/dt')

    return normalized_A, normalized_derivative
