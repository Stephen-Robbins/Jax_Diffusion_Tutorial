import matplotlib.pyplot as plt
import math

def plot_points(data,  title=None, show_axis=True):
    """
    plots 2d  points
    """
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    
    # Extract first two components and plot real data
    x, y = data[:, 0], data[:, 1]
    
    plt.scatter(x, y, alpha=1, marker='o', s=5, color='blue', label='Generated Data')

    if title:
        plt.title(title)
    
    if not show_axis:
        plt.axis('off')
    else:
        ax.legend()

    plt.tight_layout()
    plt.show()



def plot_points_over_time(data, times, title=None, show_axis=True):
    """
    Plots 2D points evolving over a range of times [0, 1] in a grid format.
    
    :param data: Array of shape [n_points, 2, n_times] containing the points to plot at each time step.
    :param times: Array of shape [n_times] containing the specific time values for each time step.
    :param title: (Optional) Title for the entire figure.
    :param show_axis: (Optional) Flag to show or hide axes.
    """
    n_times = len(times)  # Determine the number of time steps
    
    # Calculate grid size
    n_cols = math.ceil(math.sqrt(n_times))
    n_rows = math.ceil(n_times / n_cols)
    
    # Create a figure and a grid of subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    
    # Flatten the axs array for easy iteration if it's 2D
    if n_times > 1:
        axs = axs.flatten()
    
    # Loop over all time steps
    for t in range(n_times):
        ax = axs[t] if n_times > 1 else axs
        
        # Extract points for the current time step
        x, y = data[:, 0, t], data[:, 1, t]
        
        # Plot points
        ax.scatter(x, y, alpha=1, marker='o', s=5, color='blue')
        
        if not show_axis:
            ax.axis('off')
        else:
            ax.set_aspect('equal')
        
        # Individual title for each subplot showing the time
        ax.set_title(f'Time {times[t]:.2f}')
    
    # Set overall title
    if title:
        plt.suptitle(title)
    
    # Adjust layout
    plt.tight_layout(pad=3.0)
    
    # Hide unused subplots if any
    if n_times > 1:
        for idx in range(n_times, n_cols * n_rows):
            fig.delaxes(axs[idx])
    
    plt.show()

def plot_real_fake_points(real_data, fake_data, title=None, show_axis=True):
    """
    Plots the first two components of n-dimensional real and fake data points on the same plot for comparison.

    Args:
      real_data: JAX array of shape (num_samples, n) containing the real data points.
      fake_data: JAX array of shape (num_samples, n) containing the fake data points.
      title: Optional title for the plot.
      show_axis: Whether to show the axis labels and ticks (default: False).

    Returns:
      None
    """
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    
    # Extract first two components and plot real data
    x_real, y_real = real_data[:, 0], real_data[:, 1]
    
    plt.scatter(x_real, y_real, alpha=0.7, marker='o', s=5, color='blue', label='Real Data')

    # Extract first two components and plot fake data
    x_fake, y_fake = fake_data[:, 0], fake_data[:, 1]
    plt.scatter(x_fake, y_fake, alpha=0.7, marker='x', s=5, color='red', label='Generated Data')

    if title:
        plt.title(title)
    
    if not show_axis:
        plt.axis('off')
    else:
        ax.legend()

    plt.tight_layout()
    plt.show()

