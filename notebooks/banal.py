

import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

def apply_to_zeros(lst, dtype=np.int64):
    """
    Convert a list of arrays to a 2d array padded with zeros to the right.
    """
    # determine the inner max length
    inner_max_len = max(map(len, lst))

    # allocate the return array
    result = np.zeros([len(lst), inner_max_len], dtype)

    # loop over the list and fill the non-zero entries
    for i, row in enumerate(lst):
        # fill the row
        result[i,:len(row)] = row
        #for j, val in enumerate(row):
        #    result[i][j] = val
    return result

def combined_CI(dat):
    """
    Calculate a 95% confidence interval across the rows of dat.
    """
    mdat = dat.mean(0)
    edat = (dat-dat.mean(1)).std()/np.sqrt(len(dat))
    return mdat,edat*1.96


def spc(listlen=None, recalls=None, filter_ind=None, **kwargs):
    """
    Calculate the serial position curve for a list of recall lists.
    """
    if listlen is None or recalls is None:
        raise ValueError("You must specify both listlen and recalls.")

    if isinstance(recalls,list):
        # convert to padded array
        recalls = apply_to_zeros(recalls)

    if filter_ind is None:
        # make one
        filter_ind = np.ones(len(recalls), dtype=np.bool)

    # loop over serial positions to get vals
    serpos = range(1,listlen+1)
    vals = [((recalls[filter_ind]==p).sum(1)>0).mean() for p in serpos]
    return np.rec.fromarrays([serpos,vals], names='serial_pos,prec')


def prec_op(outpos=1, listlen=None, recalls=None, filter_ind=None, **kwargs):
    """
    Calculate probability of recall as a function of output position.
    """
    if listlen is None or recalls is None:
        raise ValueError("You must specify both listlen and recalls.")

    if isinstance(recalls,list):
        # convert to padded array
        recalls = apply_to_zeros(recalls)

    if filter_ind is None:
        # make one
        filter_ind = np.ones(len(recalls), dtype=np.bool)

    # loop over serial positions to get vals
    serpos = range(1,listlen+1)
    vals = [((recalls[filter_ind,outpos-1]==p)>0).mean() for p in serpos]
    return np.rec.fromarrays([serpos,vals,[outpos]*len(vals)], 
                             names='serial_pos,prec,op')


def irt_op(listlen=None, recalls=None, times=None, **kwargs):
    """
    """
    if listlen is None or recalls is None or times is None:
        raise ValueError("You must specify listlen, recalls, and times.")



def trans_fact(recs, dists):
    """
    Calculate transition factor.

    dists = -squareform(pdist(np.array([range(list_len)]).T))

    """

    # make sure recs are array
    recs = np.asanyarray(recs)

    # get lengths
    list_len = len(dists)
    nrecs = len(recs)

    # initialize containers
    tfs = np.empty(nrecs)*np.nan
    #weights = np.zeros(nrecs)

    # init poss ind
    poss_ind = np.arange(list_len)

    # loop over items
    for i in xrange(1,nrecs):
        # if current is 0, then stop
        if recs[i] == 0:
            break

        # make sure 
        # 1) current and prev valid
        # 2) not a repeat
        if ((recs[i-1]>0) and (recs[i]>0) and
            (not recs[i] in recs[:i])):
            # get possible
            ind = poss_ind[~np.in1d(poss_ind,recs[:i]-1)]
            act_ind = poss_ind[ind]==(recs[i]-1)

            if (len(ind) == 1):
                # there are not any more possible recalls other than
                # this one so we're done
                continue

            # rank them
            ranks = rankdata(dists[recs[i-1]-1][ind])
            #print ranks

            # set the tf for that transition
            tfs[i] = (ranks[act_ind]-1.)/(len(ind)-1.)

            # fiddling with weights
            #weights[i] = (ranks[act_ind])/(2.*ranks[~act_ind].mean())
            #weights[i] = np.abs(ranks[act_ind] - ranks[~act_ind]).mean()/(ranks[act_ind] - ranks[~act_ind]).std()
            #weights[i] = ranks[act_ind]/(2.*ranks[~act_ind].mean())


    return tfs #,weights


def tem_fact(listlen=None, recalls=None, filter_ind=None, **kwargs):
    """
    """
    if listlen is None or recalls is None:
        raise ValueError("You must specify both listlen and recalls.")

    if isinstance(recalls,list):
        # convert to padded array
        recalls = apply_to_zeros(recalls)

    if filter_ind is None:
        # make one
        filter_ind = np.ones(len(recalls), dtype=np.bool)

    # get the dist factor
    dists = -squareform(pdist(np.array([range(listlen)]).T))

    # get pos and neg only
    #pos_dists = dists.copy()
    #pos_dists[np.tril_indices(listlen,1)] = np.nan
    #neg_dists = dists.copy()
    #neg_dists[np.triu_indices(listlen,1)] = np.nan

    # loop over the lists
    res = []
    for i, recs in enumerate(recalls[filter_ind]):
        # get the full tfact
        tfs = trans_fact(recs, dists)

        # get the direction
        rtemp = recs.copy().astype(np.float)
        rtemp[rtemp<=0] = np.nan
        lags = np.diff(rtemp)
        lags = np.array([np.nan] + lags.tolist()).astype(np.int)

        # append the recarray of results
        res.append(np.rec.fromarrays([[i+1]*len(tfs),recs[:len(tfs)],tfs,lags], 
                                     names='list_num,rec_item,tf,lag'))
    return np.concatenate(res)


def crp(listlen=None, recalls=None, filter_ind=None, 
        allow_repeats=False, exclude_op=0, **kwargs):
    """
    Calculate a conditional response probability.

    Returns a recarray with lags, mcrp, ecrp, crpAll.
    """
    if listlen is None or recalls is None:
        raise ValueError("You must specify both listlen and recalls.")

    if isinstance(recalls,list):
        # convert to padded array
        recalls = apply_to_zeros(recalls)

    if filter_ind is None:
        # make one
        filter_ind = np.ones(len(recalls), dtype=np.bool)

    # determine possible lags
    lags = np.arange(0,2*listlen-1)-(listlen-1)

    # reset the numerator and denominator
    numer = np.zeros(len(lags),np.float64)
    denom = np.zeros(len(lags),np.float64)

    # loop over the lists
    for lis in recalls[filter_ind]:

        # loop over items in the list
        for r in np.arange(exclude_op,len(lis)-1):
            # get the items
            i = lis[r]
            j = lis[r+1]

            # see if increment, must be:
            # 1) positive serial positions (not intrusion)
            # 2) not immediate repetition
            # 3) not already recalled
            # 4) any optional conditional
            # if opt_cond is not None:
            #     opt_res = eval(opt_cond)
            # else:
            opt_res = True

            if (i>0 and j>0 and  
                i-j != 0 and     
                not np.any(np.in1d([i,j],lis[0:r])) and
                opt_res): 
                #not any(setmember1d([i,j],lis[0:r]))): 

                # increment numerator
                lag = j-i
                nInd = np.nonzero(lags==lag)[0]
                numer[nInd] = numer[nInd] + 1

                # get all possible lags
                negLag = np.arange(i-1)-(i-1)
                posLag = np.arange(i,listlen)-(i-1)
                allLag = np.union1d(negLag,posLag)

                # remove lags to previously recalled items
                if not allow_repeats:
                    recInd = np.nonzero(lis[0:r] > 0)[0]
                    recLag = lis[recInd]-i
                    goodInd = np.nonzero(~np.in1d(allLag,recLag))[0]
                    #goodInd = nonzero(~setmember1d(allLag,recLag))[0]
                    allLag = allLag[goodInd]

                # increment the denominator
                dInd = np.nonzero(np.in1d(lags,allLag))[0]
                #dInd = nonzero(setmember1d(lags,allLag))[0]
                denom[dInd] = denom[dInd] + 1

    # add in the subject's crp
    denom[denom==0] = np.nan
    crp_val = numer/denom
        
    # return the values
    return np.rec.fromarrays([lags,crp_val], names='lag,crp')


def proc_mat_subj(subj_file):
    # extract the subj
    #subj_file = 'data/ltp/stat_data_LTP265.mat'
    #bfile = os.path.splitext(os.path.basename(subj_file))[0]
    #subj = bfile[10:]

    # load the data
    x = loadmat(subj_file)['data']
    # look at sessions above 8 and up to 16
    sessions = x['session'][0,0][:,0]
    subj_num = x['subject'][0,0][0,0]
    subj = '%d'%subj_num
    min_list = 8
    if subj_num > 209:
        max_list = 16
    else:
        max_list = 14
    sess_ind = (sessions>min_list)&(sessions<=max_list)

    # lists without task switches
    info = {}
    info['recalls'] = x['recalls'][0,0][sess_ind]
    rtimes = x['times'][0,0][sess_ind]/1000.
    info['times'] = np.diff(np.hstack([np.zeros((len(rtimes),1)),rtimes])).clip(0,np.inf)
    info['listtype'] = x['pres'][0,0]['listtype'][0,0][sess_ind][:,0]
    info['distractor'] = x['pres'][0,0]['distractor'][0,0][sess_ind][:,0]
    info['final_distractor'] = x['pres'][0,0]['final_distractor'][0,0][sess_ind][:,0]
    info['task'] = x['pres'][0,0]['task'][0,0][sess_ind]
    info['subj'] = subj
    info['subjnum'] = subj_num
    info['listlen'] = x['listLength'][0,0][0,0]

    return info
