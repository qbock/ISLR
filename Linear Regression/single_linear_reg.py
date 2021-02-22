import math

class DataSet:

    def __init__(self, x, y):
        self.x = x              # Array of predictor variables
        self.y = y              # Array of response variables
        self.n = len(x)         # Number of samples


def get_slope(data, x_bar, y_bar):
    """Return the slope of the best fit line using least squares regression"""
    numerator, denominator = 0, 0
    for k in range(data.n):
        numerator += (data.x[k] - x_bar) * (data.y[k] - y_bar)
        denominator += (data.x[k] - x_bar) ** 2
    return numerator/denominator


def get_intercept(slope, x_bar, y_bar):
    """Return the intercept of the bet fit line using least squares regression"""
    return y_bar - (slope * x_bar)


def get_prediction(slope, intercept, x):
    """Returns y hat given the linear regression"""
    return slope + (intercept * x)


def get_rss(data, intercept, slope):
    """Returns the residual sum of squares"""
    rss = 0
    for k in range(data.n):
        rss += (data.y[k] - intercept - (slope * data.x[k]))**2
    return rss


def get_rse(data, rss):
    """Returns the residual standard error which is also serves as an estimate for the variance in the error"""
    return math.sqrt(rss/(data.n - 2))


def get_se(data, rse, x_bar):
    """Returns the standard error associated with the slope and with the intercept"""
    residuals = 0
    for k in range(data.n):
        residuals += (data.x[k] - x_bar)

    intercept_se = (rse ** 2) / residuals
    slope_se = (rse ** 2) * ((1/data.n) + (x_bar**2 / residuals))
    return slope_se, intercept_se


def get_t_stat(slope, slope_se):
    """Return the t-statistic associated with the data and the best fit line"""
    return slope/slope_se


def get_r_squared(rss, data, y_bar):
    """Return the r^2 statistic associated with the data and the best fit line"""
    tss = 0
    for k in range(data.n):
        tss += data.y[k] - y_bar
    return (tss - rss) / tss
