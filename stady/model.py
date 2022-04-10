import numpy
import pymc3

#Generate data with noise
number_points     = 20
true_coefficients = [10.4, 5.5]
x                 = numpy.linspace(0, 10, number_points)
noise             = numpy.random.normal(number_points)
data              = true_coefficients[0]*x + true_coefficients[1] + noise

#PRIORs:
#as sigma is unknown then we define it as a parameter:
sigma = pymc3.Uniform('sigma', 0., 100.)
#fitting the line y = a*x+b, hence the coefficient are parameters:
a     = pymc3.Uniform('a', 0., 20.)
b     = pymc3.Uniform('b', 0., 20.)

#define the model: if a, b and x are given the return value is determined, hence the model is deterministic:
@pymc.deterministic(False)
def linear_fit(a=a, b=b, x=x):
      return a*x + b

#LIKELIHOOD
#normal likelihood with observed data (with noise), model value and sigma
y = pymc3.Normal('y', linear_fit, 1.0/sigma**2, data, True)
