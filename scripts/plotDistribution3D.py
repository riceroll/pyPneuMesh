import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


matplotlib.rcParams['font.family'] = 'PT Sans'
matplotlib.rcParams['font.size'] = 16

sns.set_theme(style="darkgrid")
plt.gca().set_facecolor((0.95,0.95,0.95, 1.0))

# Random data for two groups of points
group1_x = np.random.randn(100)
group1_y = np.random.randn(100)

group2_x = np.random.randn(100) + 2  # Offset for visualization
group2_y = np.random.randn(100) + 2  # Offset for visualization

# Create a scatter plot for each group of points
plt.scatter(group1_x, group1_y, color='b', alpha=0.5, s=5, label='Selected')
plt.scatter(group2_x, group2_y, color='r', alpha=0.5, s=5,label='Removed')


plt.title('Fitness Distribution')
plt.xlabel('Fitness 0')
plt.ylabel('Fitness 1')

plt.legend(loc='lower right')  # Show the legend

plt.show()
