# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the CogMod package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np
from scipy.stats.distributions import norm, truncnorm

from samprec import _calc_p_attempts


def _trunc_norm(mean=0.0, std=1.0, lower=0.0, upper=1.0):
    """Wrapper for truncated normal."""
    a = (np.array(lower) - np.array(mean)) / np.array(std)
    b = (np.array(upper) - np.array(mean)) / np.array(std)
    return truncnorm(a, b, loc=mean, scale=std)


class SAMWrap(object):
    """Wrapper for SAM to handle range of r values."""
    def __init__(self, n_items=16, rmin=1, rmax=None,
                 params=None, scale_thresh=0.00001):
        # make sure params is at least empty dict
        if params is None:
            params = {}
            
        # handle the r min and max
        self.rmin = rmin        
        if rmax is None:
            # set to list length
            rmax = n_items
        self.rmax = rmax
                    
        # pop off the r params (defaults from fSAM)
        self.r_mu = params.get('r_mu', 4.0)
        self.r_std = params.get('r_std', 1.4)

        # determine probabilities of each r in range
        rd = _trunc_norm(mean=self.r_mu,
                         std=self.r_std,
                         lower=self.rmin-0.5,
                         upper=self.rmax+0.5)
        self.rvals = np.arange(self.rmin, self.rmax+1)
        self.p_r = np.array([rd.cdf(i+0.5)-rd.cdf(i-0.5)
                             for i in self.rvals])
        self.cdf_r = np.cumsum(self.p_r)

        # create SAM instances for each rval
        self.sams = []
        for r in self.rvals:
            # set the params
            params['r'] = r
            self.sams.append(SAM(n_items=n_items, params=params,
                                 scale_thresh=scale_thresh))

    def reset(self):
        # loop over each SAM instance
        for i in range(len(self.sams)):
            self.sams[i].reset()
        

    def present_list(self, list_type='IFR', list_def=None):
        # loop over each SAM instance
        for i in range(len(self.sams)):
            self.sams[i].present_list(list_type=list_type,
                                      list_def=list_def)
                             
    def calc_list_like(self, recalls):
        # loop over each SAM instance
        avg_likes = None
        for i in range(len(self.sams)):
            # get the likes for each recall
            likes = self.sams[i].calc_list_like(recalls)

            # scale them by p_r
            likes = np.array(likes) * self.p_r[i]

            # sum over all r
            if avg_likes is None:
                avg_likes = likes
            else:
                avg_likes += likes

        return avg_likes

    def simulate(self, nlists, list_type='IFR', list_def=None):
        # reset the model
        self.reset()

        # present the list
        self.present_list(list_type=list_type, list_def=list_def)

        # simulate lists
        recs = [self.sim_list() for i in range(nlists)]

        return recs
    
    def sim_list(self):
        # pick r from dist at random
        ind = (self.cdf_r > np.random.rand()).argmax()
        return self.sams[ind].sim_list()
        

class SAM(object):
    """Bayesian Seach of Associative Memory model"""
    default_params = {
        'a': 0.3,
        'b1': 0.5,
        'b2': None,
        'c': 0.7,
        'd': 0.01,
        'e': 0.1,
        'f1': 0.3,
        'f2': None,
        'g': 0.1,
        'r': 4,
        'r_dist': 3,
        'Kmax': 30,
        'Lmax': 4,
    }

    def __init__(self, n_items=16, params=None, scale_thresh=0.00001):
        # process the params
        self.n_items = n_items
        self.scale_thresh = scale_thresh

        # start with defaults
        p = dict(**self.default_params)
        if params is not None:
            # get provided vals
            p.update(params)
        self.params = p

        # check the possible None
        if self.params['b2'] is None:
            self.params['b2'] = self.params['b1'] / 2

        if self.params['f2'] is None:
            self.params['f2'] = self.params['f1'] / 2

        # set up the model
        self.reset()

    def reset(self):
        # init memory representations
        self.M = np.zeros((self.n_items, self.n_items))
        self.buffer = np.zeros(self.n_items)
        self.C = np.zeros(self.n_items)
        self.cur_r = 0

    def present_list(self, list_type='IFR', list_def=None):
        """Present a list to the model.

        Parameters
        ----------
        list_def: list of item_ids (currently ignored)
        list_type: {'IFR','DFR','CDFR'}
        """
        #if list_def is None:
        #    # make based on nitems
        #    list_def = range(1, self.listlen+1)

        for i in range(self.n_items):
            if i>0 and list_type[0].upper() == 'C':  # 'CDFR':
                # remove items from buffer for pre-item distractor
                for d in range(self.params['r_dist']):
                    if self.cur_r > 0:
                        self.buffer *= 1-(1/self.cur_r)
                        self.cur_r -= 1
                    else:
                        break
                    
            # make room in the buffer
            if self.cur_r >= self.params['r']:
                # we have a full buffer, so must decay
                # this does equal prob dropout
                self.buffer *= 1-(1/self.params['r'])
                self.cur_r -= 1

            # add it to the buffer
            self.buffer[i] = 1.0
            self.cur_r += 1

            # store the item-to-item associations
            self.M += np.outer(self.buffer, self.buffer)

            # store context to item associations
            self.C += self.params['a']*self.buffer

        # process the post-list distractor
        if list_type[0].upper() in ['C', 'D']:  # ['CDFR', 'DFR']
            for d in range(self.params['r_dist']):
                if self.cur_r > 0:
                    self.buffer *= 1-(1/self.cur_r)
                    self.cur_r -= 1
                else:
                    break

        # apply scaling based on learning params
        L = np.diag(np.ones(self.n_items)*self.params['c'])
        L[np.triu_indices(self.n_items, 1)] = self.params['b1']
        L[np.tril_indices(self.n_items, -1)] = self.params['b2']
        self.M *= L

        # add in baseline memory
        # it may be that this should be added to all values
        self.M[self.M<self.params['d']] = self.params['d']

        # save M and C
        self.M_save = self.M.copy()
        self.C_save = self.C.copy()
        self.buffer_save = self.buffer.copy()

    def calc_list_like(self, recalls):
        # get the saved copies of M and C
        self.M = self.M_save.copy()
        self.C = self.C_save.copy()
        self.buffer = self.buffer_save.copy()

        # convert to 0-based index
        recalls = np.atleast_1d(recalls) - 1
        
        # init the loop over items
        last_rec = None

        # var to save p_k
        p_k = np.zeros(self.params['Kmax'])

        # start with k with p(1.0) at zero
        p_k[0] = 1.0

        likes = []
        for i, rec in enumerate(recalls):
            # get rec_ind
            rec_ind = np.in1d(np.arange(self.n_items), recalls[:i])

            # first add in probability of being read out from the buffer
            # based on the probability of being in the buffer
            if i < self.cur_r:
                if rec < 0:
                    # they stopped recall, but still had items
                    # in the buffer, so zero likelihood
                    likes.append(0.0)
                    break
                
                # pull from non-recalled buffer items
                # base on probability of being in the buffer
                p_rec = self.buffer[rec]/self.buffer[~rec_ind].sum()

                # remove that item from the buffer and renormalize
                adjust_amt = 1-self.buffer[rec]
                self.buffer[rec] = 0.0
                rec_ind[rec] = True
                if adjust_amt > 0.0:
                    not_one = self.buffer < 1.0
                    self.buffer[~rec_ind & not_one] *= 1 - \
                        (adjust_amt/self.buffer[~rec_ind & not_one].sum())

                # process it and continue
                # buffer doesn't update p_k
                likes.append(p_rec)

                # do output encoding
                # increment context to item
                self.C[rec] += self.params['e']

                # self to self
                self.M[rec, rec] += self.params['g']

                # set last_rec (so last recall from buffer is used as cue)
                last_rec = rec
                continue

            if rec < 0:
                # they stopped, so calc p_stop
                # first calc p_rec for all non-recalled items
                p_nrecs = 0.0
                for nrec in np.where(~rec_ind)[0]:
                    # calc the like for the list
                    p_nrec, p_nk = self._recall_like(nrec, p_k,
                                                     last_rec=last_rec,
                                                     recalls=recalls[:i])
                    p_nrecs += p_nrec

                # p_stopping is not the sum of retrieving non-recalled items
                likes.append(1 - np.sum(p_nrecs))

                # calc the other way
                #p_stop, p_nk = self._recall_like(rec, p_k,
                #                                 recalls=recalls[:i])
                #likes.append(p_stop)

                # we're done recalling
                break
                    
            # retrieve from LTM
            # calc the like for the list
            p_rec, p_k = self._recall_like(rec, p_k,
                                           last_rec=last_rec,
                                           recalls=recalls[:i])

            # append the new rec like
            likes.append(p_rec)

            # do output encoding
            # increment context to item
            self.C[rec] += self.params['e']

            # item to item (asymmetric, but based on order of recall, not serial position)
            self.M[last_rec, rec] += self.params['f1']
            self.M[rec, last_rec] += self.params['f2']

            # self to self
            self.M[rec, rec] += self.params['g']

            # save last rec
            last_rec = rec
        return likes

    def _recall_like(self, rec, p_k, last_rec=None, recalls=None):
        if recalls is None:
            recalls = []
            
        # see if going to make item and context attempts
        if last_rec and (self.params['Lmax']>0):
            # sample some with items and context
            S = self.C.copy() + self.M[last_rec]
            p_ci_att, p_last = _calc_p_attempts(rec, S,
                                                attempts=self.params['Lmax'],
                                                recalls=recalls,
                                                scale_thresh=self.scale_thresh)
            context_att = self.params['Kmax']-self.params['Lmax']
        else:
            context_att = self.params['Kmax']
            p_ci_att = []
            p_last = 1.0

        # attempts with just context
        if context_att > 0:
            S = self.C.copy()
            p_c_att, p_last = _calc_p_attempts(rec, S,
                                               attempts=context_att,
                                               recalls=recalls, 
                                               p_start=p_last,
                                               scale_thresh=self.scale_thresh)

            # concat all attempts
            p_att = np.concatenate([p_ci_att, p_c_att])
        else:
            # just go with ci attempts
            p_att = p_ci_att

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
        recs = [self.sim_list() for i in range(nlists)]

        return recs
    
    def sim_list(self):
        # save starting C and M (do be restored after simulation)
        self.M = self.M_save.copy()
        self.C = self.C_save.copy()
        self.buffer = self.buffer_save.copy()
        
        # init the loop over items
        last_rec = None

        # var to save p_k
        p_k = np.zeros(self.params['Kmax'])

        # start with k with p(1.0) at zero
        p_k[0] = 1.0

        # init recalls
        recalls = []
        rec_ind = np.zeros(self.n_items, dtype=np.bool)
        for i in range(self.n_items):
            # first recall with buffer
            if i < self.cur_r:
                # pull from non-recalled buffer items
                # base on probability of being in the buffer
                p_recs = self.buffer[~rec_ind]/self.buffer[~rec_ind].sum()
                cdfs = np.cumsum(p_recs)

                # pick one at random
                ind = (cdfs > np.random.rand()).argmax()
                rec = np.where(~rec_ind)[0][ind]
                recalls.append(rec)

                # set the last_rec
                last_rec = rec
                rec_ind[rec] = True

                # remove that item from the buffer and renormalize
                adjust_amt = 1-self.buffer[rec]
                self.buffer[rec] = 0.0
                if adjust_amt > 0:
                    not_one = self.buffer < 1.0
                    self.buffer[~rec_ind & not_one] *= 1 - \
                        (adjust_amt/self.buffer[~rec_ind & not_one].sum())

                # do output encoding
                # increment context to item
                self.C[rec] += self.params['e']

                # self to self
                self.M[rec, rec] += self.params['g']

                # keep going
                continue

            # recall from LTM
            # loop over not-recalled items to get likes
            p_recs = []
            p_nks = []
            recs = []
            for nrec in np.where(~rec_ind)[0]:
                # calc the like for the list
                p_nrec, p_nk = self._recall_like(nrec, p_k,
                                                 last_rec=last_rec,
                                                 recalls=recalls[:i])

                p_recs.append(p_nrec)
                p_nks.append(p_nk)
                recs.append(nrec)

            # append p_stop
            recs.append(-1)
            cdfs = np.concatenate([np.cumsum(p_recs), [1.0]])
            
            #p_stop, p_nk = self._recall_like(-1, p_k,
            #                                 recalls=recalls[:i])
            #p_recs.append(p_stop)

            # normalize to fix approx
            #p_recs = np.array(p_recs)
            #p_recs = p_recs/p_recs.sum()
            #cdfs = np.cumsum(p_recs)
            #cdfs = np.concatenate([np.cumsum(p_recs), [1.0]])
            
            # pick a recall at random
            ind = (cdfs > np.random.rand()).argmax()
            rec = recs[ind]
            recalls.append(rec)
            if rec < 0:
                # all done
                break

            # set the last_rec and p_k
            last_rec = rec
            p_k = p_nks[ind]
            rec_ind[rec] = True

            # do output encoding
            # increment context to item
            self.C[rec] += self.params['e']

            # item to item (asymmetric, but based on order of recall, not serial position)
            self.M[last_rec, rec] += self.params['f1']
            self.M[rec, last_rec] += self.params['f2']

            # self to self
            self.M[rec, rec] += self.params['g']

        # add one to the returns
        return np.atleast_1d(recalls)+1


    
