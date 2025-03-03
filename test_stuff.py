import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create figure and axis
fig, ax = plt.subplots()

# Plot the data
ax.plot(x, y1, label="sin(x)")
ax.plot(x, y2, label="cos(x)")

# Label the curves at specific positions
ax.text(8, np.sin(8), "sin(x)", fontsize=12, color="blue", verticalalignment='bottom')
ax.text(8, np.cos(8), "cos(x)", fontsize=12, color="orange", verticalalignment='bottom')

# Show the plot
plt.show()