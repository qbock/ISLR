import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from single_linear_reg import *

# Generate fake data with linear form
slope = 3
intercept = 4
x = np.linspace(0, 100, 1000)
y = np.zeros(1000)

count = 0
for k in x:
    noise = stats.norm.rvs(0, 20, 1)
    y[count] = intercept + (k * slope) + noise
    count += 1

data = DataSet(x, y)

# Get slope, intercept, and stats about the data

x_bar, y_bar = np.mean(x),np.mean(y)
slope = get_slope(data, x_bar, y_bar)
intercept = get_intercept(slope, x_bar, y_bar)
rss = get_rss(data, intercept, slope)
rse = get_rse(data, rss)
slope_se, intercept_se = get_se(data, rse, x_bar)
t_stat = get_t_stat(slope, slope_se)
r_squared = get_r_squared(rss, data, y_bar)

print(f'Slope: {slope} \nIntercept: {intercept} \nRSS: {rss} \nRSE: {rse} \nSlope Standard Error: {slope_se}')
print(f'Intercept Standard Error: {intercept_se} \nT Statistic: {t_stat} \nR Squared: {r_squared}')


def f(t):
    return intercept + (t*slope)


plt.plot(x, y, 'ro', x, f(x), "b-", markersize=1)
plt.show()

