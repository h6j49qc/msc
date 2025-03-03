import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Sample data
t = np.linspace(0, 10, 100)
primary_data = np.exp(-t)  # Decay function
secondary_data = primary_data / primary_data.max()  # Normalized (0 to 1)

# Create figure and axis
fig, ax1 = plt.subplots(figsize=(8, 5))

# Primary axis
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Concentration (M)")
ax1.plot(t, primary_data, label="Concentration", color="blue")
ax1.legend(loc="upper right")

# Secondary axis (Percentage)
ax2 = ax1.twinx()
ax2.set_ylabel("Proportion (%)", color="green")
ax2.plot(t, secondary_data, label="Proportion", color="green", linestyle="dashed")

# Set secondary y-axis format to percentage
ax2.set_ylim(0, 1)  # Ensures 0% to 100% range
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))

ax2.legend(loc="lower right")

# Apply grid on x-axis only
plt.grid(axis="x")

plt.title("Example: Primary and Secondary Y-Axis with Percentage Scale")
plt.show()
