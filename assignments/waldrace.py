# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the CogMod package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np
from scipy.stats.distributions import invgauss, norm


def trdm_like(choice, t, v, alpha, theta, sig, rho=0):
    """Timed Racing Diffusion Model (TRDM)
    
    rho - Scaler between 0 and 1 for the tradeoff between the timer
    giving rise to chance decisions (when rho=0) and the timer giving
    rise to the decision based on the ratio of the CDF for the choice
    to the CDF of all choices (when rho=1), approximating the
    proportion of evidence for that choice.

    """
    d_choice, d_timer = _trdm_density(choice, t, v,
                                      alpha, theta, sig, rho=rho)
    return d_choice + d_timer

def _trdm_density(choice, t, v, alpha, theta, sig, rho=0):
    # make sure params are arrays
    choice = np.atleast_1d(choice)
    t = np.atleast_1d(t)
    v = np.atleast_1d(v)
    
    # set n_choice based on the drift rates
    # (last drift rate is for timer)
    n_race = len(v)
    n_choice = n_race-1

    # if only supplying same vals for other params
    # replicate them
    alpha = np.atleast_1d(alpha)
    if len(alpha) < n_race:
        alpha = alpha.repeat(n_race)

    theta = np.atleast_1d(theta)
    if len(theta) < n_race:
        theta = theta.repeat(n_race)

    sig = np.atleast_1d(sig)
    if len(sig) < n_race:
        sig = sig.repeat(n_race)    
    
    # make the choice values zero-based
    choice = choice-1
    
    # pick the unique race indices
    uniq_choice = np.arange(n_race)
    
    # initialize density values to zeros
    d_choice = np.zeros(len(t))
    d_timer = np.zeros(len(t))
    
    # fix the drift rates as needed
    bad_ind = (v<1e-100)|(v==np.inf)|(v==np.nan)
    v[bad_ind] = 1e-100
    
    # calc 1-CDF for each choice 
    # (probability they have not made that choice, yet)
    x = t[:, np.newaxis]-theta
    mu = alpha/v
    lamb = (alpha/sig)**2
    not_sel = 1-invgauss((mu/lamb)[np.newaxis]).cdf(x/lamb[np.newaxis])

    # process the timer's pdf
    f_timer = invgauss(mu[-1]/lamb[-1]).pdf(x[:,-1]/lamb[-1])*(1/lamb[-1])

    # process non-responses
    # PBS: Must double-check this
    ind = (choice==-1) & np.all((x > 0), axis=1)
    if ind.sum() > 0:
        # calc p not sel for all choices
        # putting everything in choice, not timer
        d_choice[ind] = np.product(not_sel[ind], axis=1)

    # loop over each choice
    for j in range(n_choice):
        # pick the trials with that choice
        ind = (choice==j) & (x[:, j] > 0)
        
        if ind.sum() == 0:
            # there weren't any trials with that choice, so skip it
            continue

        # process the selected choice
        # calculate the probability of making that response
        # at the specified rt
        f_sel = invgauss(mu[j]/lamb[j]).pdf(x[ind][:, j]/lamb[j])*(1/lamb[j])
        
        # get the p not selected for non-selected choices (includes timer)
        p_term = np.product(not_sel[ind][:, uniq_choice!=j],
                            axis=1)

        # timer choice based on ratio of cdfs of choice accumulators (or chance)
        if rho > 0:
            # mixture of chance and probability of accumulator being ahead
            # pick the times for the non-selected options
            nctimes = x[ind][:, uniq_choice!=j][:, :-1]

            # pick the times for the selected options
            ctimes = x[ind][:, uniq_choice==j]

            # calculate the differences in means at those times
            mu_diff = (v[uniq_choice!=j][:-1]*nctimes - 
                       v[j]*ctimes)

            # calc sum of variances (then take sqrt) at those times
            std_sum = np.sqrt((sig[uniq_choice!=j][:-1]*np.sqrt(nctimes))**2 + \
                              (sig[j]*np.sqrt(ctimes))**2)

            # CDF at 0 tells the probability of being ahead for each choice
            # Product is the probability of being ahead of all other choices
            p_ahead = np.product(norm(loc=mu_diff, scale=std_sum).cdf(0), axis=1)
            
            # combine with chance performance based on rho
            p_choice = rho*p_ahead + (1-rho) * 1./n_choice
        else:
            # pick by chance
            p_choice = 1./n_choice

        # calculate the density for that choice
        # 1) p(choice) * p(other choices not done) * p(timer not done)
        # 2) p(choice) * p(timer going off) * p(all choices not done)        
        d_choice[ind] = (f_sel * p_term)
        d_timer[ind] = (p_choice * f_timer[ind] * np.product(not_sel[ind, :-1], axis=1))
    return d_choice, d_timer


def trdm_gen(v, alpha, theta, sig, rho=0,
             dt=0.001, max_time=5.0, nsamp=1000):

    # get choice counts
    n_race = len(v)
    n_choice = n_race - 1
    
    # generate time range
    trange = np.arange(dt, max_time+dt, dt)

    # calc cdf of each choice
    rts = np.concatenate([trange]*n_choice + [[-1.]])
    ntimes = len(trange)
    choices = np.ones(len(rts), dtype=np.int)
    for i in range(1, n_choice):
        choices[i*ntimes:i*ntimes+ntimes] = i+1
    choices[-1] = 0
    timer_lookup = np.array([0]*(len(rts)-1) + [1]*(len(rts)-1) + [-1])
    rts_lookup = np.concatenate([rts[:-1], rts[:-1], [-1]])
    choices_lookup = np.concatenate([choices[:-1], choices[:-1], [0]])

    # evaluate the likelihoods for choices and times
    d_choice, d_timer = _trdm_density(choices[:-1], rts[:-1],
                                      v, alpha, theta, sig, rho=rho)

    # stack the densities for choices or timer
    likes = np.concatenate([d_choice, d_timer])

    # generate desired responses
    # calc cdfs
    cdfs = np.concatenate([(likes*dt).cumsum(), [1.0]])

    # draw uniform rand numbers to determine choices and rts
    inds = [(cdfs > np.random.rand()).argmax()
            for i in range(nsamp)]

    return choices_lookup[inds], rts_lookup[inds], timer_lookup[inds]

    
if __name__ == '__main__':
    t = np.array([.5, .6, .7, .8])
    choice = np.array([2, 1, 1, 1])
    v = np.array([5.5, 2.5, 0.5])
    alpha = np.array([2.0])
    theta = np.array([.2])
    sig = np.array([1.0])

    trdm_like(t, choice, v, alpha, theta, sig, rho=.5)
