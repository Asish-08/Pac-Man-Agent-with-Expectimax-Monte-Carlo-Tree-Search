import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro

# Generate some example data (replace this with your dataset)
data = [0.49602183, 0.42711368, 0.44915869, 0.48938661, 0.47410606, 0.51424642, 0.50815884, 0.51744358, 0.49660348, 0.57913047,
0.62285512, 0.5308816, 0.66075058, 0.61961095, 0.62975977, 0.44994393, 0.7014689, 0.7503477, 0.59846672, 0.56859008,
0.62910785, 0.84311231, 0.48803497, 0.69556573, 0.66977742, 0.55436494, 0.78429556, 0.55390514, 0.51365292, 0.60860025]

# Plot histogram
plt.hist(data, bins=30, density=True, alpha=0.6, color='g')

# Fit a normal distribution to the data
mu, std = norm.fit(data)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)

# Test for normality using Shapiro-Wilk test
shapiro_test_statistic, shapiro_p_value = shapiro(data)
is_normal = shapiro_p_value > 0.05
if is_normal:
    plt.title('Normally Distributed (p-value={:.4f})'.format(shapiro_p_value))
else:
    plt.title('Not Normally Distributed (p-value={:.4f})'.format(shapiro_p_value))

plt.grid(True)
plt.show()
