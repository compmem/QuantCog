# some useful conversions pulled from RunDEMC

# scipy provides distributions
import scipy.stats.distributions as dists

def uniform(lower=0.0, upper=1.0):
    return dists.uniform(loc=lower, scale=upper - lower)


def normal(mean=0.0, std=1.0):
    return dists.norm(loc=mean, scale=std)


def beta(alpha=.5, beta=.5):
    return dists.beta(alpha, beta)


def gamma(alpha=1.0, beta=1.0):
    """
    alpha = k
    beta = 1/theta
    """
    return dists.gamma(alpha, scale=1. / beta)


def invgamma(alpha=1.0, beta=1.0):
    """
    """
    return dists.invgamma(alpha, scale=beta)


def exp(lam=1.0):
    return dists.expon(scale=1. / lam)


def poisson(lam=1.0):
    return dists.poisson(mu=lam)


def laplace(loc=0.0, diversity=1.0):
    return dists.laplace(loc=loc, scale=diversity)


def students_t(mean=0, std=1.0, df=1.0):
    return dists.t(df=df, loc=mean, scale=std)


def noncentral_t(mean=0, std=1.0, df=1.0, nc=0.0):
    return dists.nct(df=df, nc=nc, loc=mean, scale=std)


def halfcauchy(loc=0.0, scale=1.0):
    return dists.halfcauchy(loc=loc, scale=scale)

