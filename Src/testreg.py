from numpy import*
from pymcmc.regtools import BayesRegression

X=random.randn(100,3)
beta = random.randn(3)

y=dot(X,beta) + 0.3 * random.randn(100)

breg=BayesRegression(y,X,prior=['normal_inverted_gamma', zeros(3), eye(3)*0.1, 10, 0.01])
print breg.log_marginal_likelihood()

breg=BayesRegression(y,X)
print breg.log_marginal_likelihood()

breg=BayesRegression(y,X, prior = ['g_prior', 0.0, 100.0])
print breg.log_marginal_likelihood()
