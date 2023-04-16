import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import subprocess
# Your data points
data = pd.read_csv('../dataset_building.csv')
data_points = list(data.iloc[250:270,0].values)



# Generate random data
x = np.arange(0, 15*20, 15)
y = np.array(data_points)
y_2 = np.array(data_points)*np.random.uniform(0.8, 0.90, size=20)
# Create figure and axis objects
fig, ax = plt.subplots()

# Set initial plot properties
ax.set_xlim(0, 15*20)
ax.set_ylim(0, max(data_points)+20)
ax.set_xlabel("Time (min)")
ax.set_ylabel("Energy (kW)")
# line, = ax.plot([], [], label="Standard Hvac")
# line_2, = ax.plot([], [], label="W/ WATTAI")
line, = ax.plot([], [], label="Standard Hvac", linewidth=2, linestyle='-', color='r')
line_2, = ax.plot([], [], label="W/ WATTAI", linewidth=1.5, linestyle='--', color='b')

ax.legend()

# Define the function to update the plot
def update_plot(i):
    # Add a new data point
    line.set_data(x[:i], y[:i])
    line_2.set_data(x[:i], y_2[:i])
    return line,line_2

# Create and save the animation
fps = 1/1.5  # Frames per second
duration = 30  # Duration of the video in seconds
num_frames = int(fps * duration)
interval = 1.5  # Interval between each data point in seconds
frames = [i for i in range(0, num_frames, int(fps * interval))]
ani = FuncAnimation(fig, update_plot, frames=frames, blit=True)
ani.save("videos/plot_video.mp4", fps=fps, extra_args=["-vcodec", "libx264"])

# Convert the video to GIF using ffmpeg
subprocess.call("ffmpeg -i videos/plot_video.mp4 -vf scale=320:-1 -r 10 plot_video.gif", shell=True)





# plt.figure()
# plt.plot(data.iloc[250:270,2])
# plt.plot(data.iloc[250:270,3])
# plt.plot(data.iloc[250:270,4])

# plt.figure()
# plt.plot(data.iloc[250:270,0])
# plt.plot(data.iloc[250:270,0]*0.83)
# plt.show()



