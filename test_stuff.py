import matplotlib.pyplot as plt

# Create a figure and axis
fig, ax = plt.subplots()

# Plot some dummy data
ax.plot([1, 2, 3], [4, 5, 6])

# Add a title with a border
ax.set_title("My Plot Title", bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

# Show the plot
plt.show()
