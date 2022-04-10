from numpy             import polyfit
from matplotlib.pyplot import figure, plot, show, legend
import pymc3
import model

#Define MCMC:
D = pymc3.MCMC(model, 'pickle')

#Sample MCMC: 10000 iterations, burn-in period is 1000
D.sample(10000,1000)


#compute chi-squared fitting for comparison:
chisq_result = polyfit(model.x, model.data, 1)

#print the results:
print ("\n\nResult of chi-square result: a= %f, b= %f" % (chisq_result[0], chisq_result[1]))
print ("\nResult of Bayesian analysis: a= %f, b= %f" % (D.a.value, D.b.value))
print ("\nThe real coefficients are:   a= %f, b= %f\n" %(model.true_coefficients[0], model.true_coefficients[1]))

#plot graphs from MCMC:
pymc3.Matplot.plot(D)

#plot noised data, true line and two fitted lines (bayes and chi-squared):
figure()
plot(model.x, model.data,'+','')
plot(model.x, D.a.value * model.x + D.b.value,'g','Bayes')
plot(model.x, chisq_result[0] * model.x + chisq_result[1],'r', 'Chi-squared')
plot(model.x, model.true_coefficients[0] * model.x + model.true_coefficients[1], 'k', 'Data')
legend()
show()
