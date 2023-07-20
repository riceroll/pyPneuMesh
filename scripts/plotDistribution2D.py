import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white")

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Helvetica Neue'
matplotlib.rcParams['font.size'] = 7

# plt.gca().set_facecolor((0.95,0.95,0.95, 1.0))

plt.tick_params(left="on", bottom="on", length=4, width=1.0)

# Random data for two groups of points
group1_x = np.random.randn(100)
group1_y = np.random.randn(100)

group2_x = np.random.randn(100) + 2  # Offset for visualization
group2_y = np.random.randn(100) + 2  # Offset for visualization

# Create a scatter plot for each group of points
plt.scatter(group1_x, group1_y, color='b', alpha=0.5, s=5, label='Selected')
plt.scatter(group2_x, group2_y, color='r', alpha=0.5, s=5,label='Removed')


# Change the frame width
ax = plt.gca()  # Get the current axes
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.0)  # Set the line width to 2.0


plt.title('Fitness Distribution')
plt.xlabel('Fitness 0')
plt.ylabel('Fitness 1')

plt.legend(frameon =False, loc='lower right')  # Show the legend

plt.show()
