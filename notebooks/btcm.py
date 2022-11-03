# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the CogMod package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##


import numpy as np
from numba import njit

from samprec import _calc_p_attempts

def l1_norm(dat):
    """
    l1_norm along the first dimension of a data array.
    """
    denom = np.abs(dat).sum(axis=0)[np.newaxis]
    denom[denom == 0.0] = 1.0
    return dat/denom.repeat(len(dat), axis=0)


def l2_norm(dat):
    """
    l2_norm along the first dimension of a data array.
    """
    denom = np.sqrt(np.power(dat, 2).sum(axis=0))[np.newaxis]
    denom[denom == 0.0] = 1.0
    return dat/denom.repeat(len(dat), axis=0)


class TCM(object):
    """Temporal Context Model
    """
    default_params = {
        # assoc
        'rho': .5,
        'rho_dist': None,
        'rho_ret': None,
        'beta': .5,
        'phi': 2.0,
        'gamma': .5,
        'lambda': 0.5,
        'alpha': 1.0,

        # retrieval (recall)
        'sigma_base': 0.0,
        'sigma_exp': 4.0,
        'Kmax': 30,
        'tau': 1.0,
        'xi': .0001

    }

    def __init__(self, listlen=16, nitems=None, params=None,
                 scale_thresh=0.00001):
        """
        """
        # save the nitems
        self.listlen = listlen
        if nitems is None:
            nitems = (listlen*2)+3
        self.nitems = nitems
        self.items = np.eye(nitems)
        self.scale_thresh = scale_thresh

        # process the params
        # start with defaults
        p = dict(**self.default_params)
        if params is not None:
            # get provided vals
            p.update(params)
        self.params = p

        # check the rho values
        if self.params['rho_dist'] == None:
            self.params['rho_dist'] = self.params['rho']
        if self.params['rho_ret'] == None:
            self.params['rho_ret'] = self.params['rho']

        # set phi_decay from rho
        self.params['phi_decay'] = -np.log(self.params['rho'])

        # set up the model
        self.reset()

    def reset(self):
        # allocate for all matrices and vectors
        self.M = np.zeros((self.nitems, self.nitems))
        self.f1 = np.zeros(self.nitems)
        self.f0 = np.zeros(self.nitems)
        self.t1 = np.zeros(self.nitems)
        self.t0 = np.zeros(self.nitems)

        # set up t0 with init item
        self.t0[0] = 1.0

        # set up list context
        self.lc_ind = -1
        self.t0[self.lc_ind] = self.params['lambda']

        # normalize it (not including list context unit)
        self.t0[:-1] = l2_norm(self.t0[:-1])

        # set current distractor ind
        self.cur_dist = -2
        self.cur_pres = 0

        pass

    def _present_item(self, i, rho, alpha=0.0):

        # pick the item
        self.f1 = self.items[i]

        # calc the new t
        tIN = l2_norm(self.params['beta']*self.f1 +
                      (1-self.params['beta'])*np.dot(self.f1, self.M))
        #self.t1 = rho*self.t0 + (1-rho)*tIN
        self.t1 = rho*self.t0 + tIN

        # reset the list context unit
        self.t1[self.lc_ind] = self.params['lambda']

        # make unit length (not including list context unit)
        self.t1[:-1] = l2_norm(self.t1[:-1])

        # update M
        if alpha > 0.0:
            phi = alpha + (self.params['phi'] *
                           np.exp(-self.params['phi_decay'] *
                                  (self.cur_pres)))
            self.M += phi * np.outer(self.f1, self.t0)

        # update cur_pres
        self.cur_pres += 1

        # set the latest item/context as the old
        self.f0 = self.f1
        self.t0 = self.t1

    def present_list(self, list_def=None, list_type='IFR'):
        """Present a list to the model.

        Parameters
        ----------
        list_def: list of item_ids
        list_type: {'IFR','DFR','CDFR'}
        """
        if list_def is None:
            # make based on nitems
            list_def = range(1, self.listlen+1)

        # loop over items
        for i in list_def:
            if list_type[0].upper() == 'C':  # 'CDFR':
                # present premath
                self._present_item(self.cur_dist,
                                   self.params['rho_dist'],
                                   alpha=self.params['alpha'])
                self.cur_dist -= 1

            # present the item
            self._present_item(i, self.params['rho'],
                               alpha=self.params['alpha'])

        # see if postmath
        if list_type[0].upper() in ['C', 'D']:  # ['CDFR', 'DFR']
            self._present_item(self.cur_dist,
                               self.params['rho_dist'],
                               alpha=self.params['alpha'])
            self.cur_dist -= 1

        # save current context
        self.t_save = self.t0.copy()

    @property
    def strengths(self):
        #return (self.params['gamma']*np.dot(self.M, self.t0) +
        #        (1-self.params['gamma'])*self.t0)
        return (np.dot(self.M, self.t0) +
                (self.params['gamma']*self.t0))

    def calc_list_like(self, recalls):
        # reset context
        self.t0 = self.t_save.copy()

        # convert to 0-based index
        recalls = np.atleast_1d(recalls) - 1
        
        # var to save p_k
        p_k = np.zeros(self.params['Kmax'])

        # start with k with p(1.0) at zero
        p_k[0] = 1.0

        likes = []
        for i, rec in enumerate(recalls):
            # get rec_ind
            rec_ind = np.in1d(np.arange(self.listlen), recalls[:i])

            if rec < 0:
                # they stopped, so calc p_stop
                # first calc p_rec for all non-recalled items
                p_nrecs = 0.0
                for nrec in np.where(~rec_ind)[0]:
                    # calc the like for the list
                    p_nrec, p_nk = self._recall_like(nrec, p_k,
                                                     recalls=recalls[:i])
                    p_nrecs += p_nrec
                p_stop = 1-np.sum(p_nrecs)
                
                # calc the other way
                #p_stop, p_nk = self._recall_like(rec, p_k,
                #                                 recalls=recalls[:i])
                
                # p_stopping is not the sum of retrieving non-recalled items
                #likes.append(1 - np.sum(p_nrecs))
                likes.append(p_stop)

                # we're done recalling
                break
                    
            # retrieve from LTM
            # calc the like for the list
            p_rec, p_k = self._recall_like(rec, p_k,
                                           recalls=recalls[:i])

            # append the new rec like
            likes.append(p_rec)

            # do output encoding and move to next item
            self._present_item(rec+1, self.params['rho_ret'], alpha=0.0)
            
        return likes

    def _recall_like(self, rec, p_k, recalls=None):
        if recalls is None:
            recalls = []

        # set up the number of attempts
        context_att = self.params['Kmax']
        p_last = 1.0

        # attempts with just context
        S = self.strengths[1:self.listlen+1]**self.params['tau']
        #CV = S[S>self.params['xi']].std()/S[S>self.params['xi']].mean()
        #S *= CV
        #print(CV)
        #S = S**CV        
        p_att, p_last = _calc_p_attempts(rec, S,
                                         recalls=recalls, 
                                         attempts=context_att,
                                         p_start=p_last,
                                         scale_thresh=self.scale_thresh)

        # do weighted combo of attempts
        new_p_k = np.zeros(self.params['Kmax'])
        if rec < 0:
            # adjust p_att
            p_att = np.concatenate([p_att[1:], [p_last]])
            
            # we're testing stopping, so just pick end values
            for k, p in enumerate(p_k):
                new_p_k[k] += \
                    p * p_att[self.params['Kmax']-k-1]
        else:
            # do weighted combo of all different p_k
            for k, p in enumerate(p_k):
                new_p_k[k:self.params['Kmax']] += \
                    p * p_att[:self.params['Kmax']-k]

        # p_rec is just sum over new_p_k
        p_rec = new_p_k.sum()

        # set new p_k
        if rec < 0:
            # we've stopped, so we have maxed out k
            p_k = np.zeros(self.params['Kmax'])
        else:
            # normalize new p_k for next iteration
            p_k = new_p_k/new_p_k.sum()

        return p_rec, p_k

    def simulate(self, nlists, list_type='IFR', list_def=None):
        # reset the model
        self.reset()

        # present the list
        self.present_list(list_type=list_type, list_def=list_def)

        # simulate lists
        #recs = [self.sim_list() for i in range(nlists)]
        recs = [_tcm_sim_recs(self.M, self.items, self.t_save.copy(),
                              self.listlen, self.params['beta'],
                              self.params['rho_ret'], self.params['gamma'],
                              self.params['tau'], self.params['lambda'],
                              self.params['Kmax'])
                for i in range(nlists)]

        return recs
    
    def sim_list(self):
        # reset context
        self.t0 = self.t_save.copy()

        # var to save p_k
        p_k = np.zeros(self.params['Kmax'])

        # start with k with p(1.0) at zero
        p_k[0] = 1.0

        # init recalls
        recalls = []
        rec_ind = np.zeros(self.listlen, dtype=np.bool)
        for i in range(self.listlen):
            # recall from LTM
            # loop over not-recalled items to get likes
            p_recs = []
            p_nks = []
            recs = []
            for nrec in np.where(~rec_ind)[0]:
                # calc the like for the list
                p_nrec, p_nk = self._recall_like(nrec, p_k,
                                                 recalls=recalls[:i])

                p_recs.append(p_nrec)
                p_nks.append(p_nk)
                recs.append(nrec)

            # append p_stop
            recs.append(-1)
            p_stop, p_nk = self._recall_like(-1, p_k,
                                             recalls=recalls[:i])
            p_recs.append(p_stop)

            # normalize to fix approx
            p_recs = np.array(p_recs)
            p_recs = p_recs/p_recs.sum()
            #p_recs[:-1] = p_recs[:-1]*(1-p_recs[-1])/(p_recs[:-1]).sum()
            #cdfs = np.concatenate([np.cumsum(p_recs), [1.0]])
            cdfs = np.cumsum(p_recs)
            
            # pick a recall at random
            ind = (cdfs > np.random.rand()).argmax()
            rec = recs[ind]
            recalls.append(rec)
            if rec < 0:
                # all done
                break

            # set the p_k
            p_k = p_nks[ind]
            rec_ind[rec] = True

            # do output encoding (item processing)
            self._present_item(rec+1, self.params['rho_ret'], alpha=0.0)
        
        # add one to the returns
        return np.atleast_1d(recalls)+1


@njit
def _tcm_sim_recs(M, items, t0, listlen, beta, rho, gamma, tau, lamb, Kmax):
    # do recalls
    recalls = np.zeros(listlen)
    rec_ind = np.zeros(listlen, dtype=np.bool_)
    k = 0
    for i in range(listlen):
        # get strength
        S = (np.dot(M, t0) + (gamma*t0))[1:listlen+1]**tau
        p_s = (S/S.sum())
        p_r = (1-np.exp(-S))
        samp_ind = rec_ind.copy()
        for l in range(Kmax):
            # sample an item
            ind = (np.cumsum(p_s) > np.random.rand()).argmax()

            if samp_ind[ind]:
                # we've already sampled it, so skip
                k += 1
                if k >= Kmax:
                    break
                continue

            # we've sampled it now
            samp_ind[ind] = True

            # see if recover
            if np.random.rand() < p_r[ind]:
                # we recover it
                rec = ind
                recalls[i] = rec + 1
                rec_ind[rec] = True

                # update context and start over
                # pick the item
                f1 = items[rec+1]

                # calc the new t
                tIN = beta*f1 + (1-beta)*np.dot(f1, M)
                tIN /= np.sqrt((tIN**2).sum())
                t1 = rho*t0 + tIN

                # reset the list context unit
                t1[-1] = lamb

                # make unit length (not including list context unit)
                t1[:-1] = t1[:-1]/np.sqrt((t1[:-1]**2).sum())

                # replace new context
                t0 = t1

                break
            else:
                # we failed
                k += 1
                if k >= Kmax:
                    break

        if k >= Kmax:
            # enough failures
            break
    
    return recalls



if __name__ == "__main__":

    # set up items
    listlen = 16
    nlists = 1000

    params = {
        # assoc
        'rho': .238,
        'rho_dist': .070,
        'rho_ret': None,
        'beta': .965,
        'phi': 1.201,
        'gamma': .050,
        'lambda': .349,

        # retrieval
        'sigma_base': 0.0,
        'sigma_exp': 11.647,
        'tau': 1.306}

    tcm = TCM(listlen, params=params)

    recalls = tcm.simulate(nlists=nlists,
                           list_def=range(1, listlen+1),
                           list_type='DFR')

    #ll = [tcm.calc_list_like(recs) for recs in recalls]

    recs = np.zeros((len(recalls), listlen))
    for i, r in enumerate(recalls):
        recs[i, :len(r)] = r
