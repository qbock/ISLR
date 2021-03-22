import math
import pandas as pd

def get_slope(df, x_bar, y_bar):
    """Return the slope of the best fit line using least squares regression"""
    numerator, denominator = 0, 0
    x, y = df['x'], df['y']
    for k in range(len(df.index)):
        numerator += (x[k] - x_bar) * (y[k] - y_bar)
        denominator += (df.x[k] - x_bar) ** 2
    return numerator/denominator


def get_intercept(slope, x_bar, y_bar):
    """Return the intercept of the bet fit line using least squares regression"""
    return y_bar - (slope * x_bar)


def get_prediction(slope, intercept, x):
    """Returns y hat given the linear regression"""
    return slope + (intercept * x)


def get_rss(df, intercept, slope):
    """Returns the residual sum of squares"""
    rss = 0
    x, y = df['x'], df['y']
    for k in range(len(df.index)):
        rss += (y[k] - intercept - (slope * x[k]))**2
    return rss


def get_rse(df, rss):
    """Returns the residual standard error which is also serves as an estimate for the variance in the error"""
    return math.sqrt(rss/(len(df.index) - 2))


def get_se(df, rse, x_bar):
    """Returns the standard error associated with the slope and with the intercept"""
    residuals = 0
    x, y = df['x'], df['y']
    for k in range(len(df.index)):
        residuals += (x[k] - x_bar)

    intercept_se = (rse ** 2) / residuals
    slope_se = (rse ** 2) * ((1/len(df.index)) + (x_bar**2 / residuals))
    return slope_se, intercept_se


def get_t_stat(slope, slope_se):
    """Return the t-statistic associated with the data and the best fit line"""
    return slope/slope_se


def get_r_squared(rss, df, y_bar):
    """Return the r^2 statistic associated with the data and the best fit line"""
    tss = 0
    y = df['y']
    for k in range(len(df.index)):
        tss += y[k] - y_bar
    return (tss - rss) / tss
