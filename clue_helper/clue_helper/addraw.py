'''
This is addraw, a convenience module for the production run of our `clue` paper.
'''
import numpy as np
import pickle as pkl
import scanpy as sc


mountpoint = '/data/clue/'

def set_mountpoint(path):
    global mountpoint
    global prefix_adts
    global prefix_mrna
    global prefix_comb
    
    mountpoint = path
    prefix_adts = mountpoint + 'prod/adts/'
    prefix_mrna = mountpoint + 'prod/mrna/'
    prefix_comb = mountpoint + 'prod/comb/'
    return

def get_raw_adts(obs_names):
    path = prefix_adts + 'pkls/concat_adts.pkl'
    
    with open(path,'rb') as file:
        return pkl.load(file)[obs_names, :]

def get_raw_mrna(obs_names):
    path = prefix_mrna + 'pkls/wells_sng_covars.pkl'
    
    with open(path,'rb') as file:
        wells = pkl.load(file)
    
    for well in wells:
        wells[well]['adata'].obs_names = [i[:16] + '-%s' % well for i in wells[well]['adata'].obs_names]
    
    return wells[0]['adata'].concatenate(*[wells[i]['adata'] for  i in range(1, 12)])[obs_names,:]
    
def clr_normalize_column(x):
    normed_column = np.log1p((x) / (np.exp(sum(np.log1p((x)[x > 0 ])) / len(x + 1))))
    return normed_column

def clr_normalize(x):
    normed_matrix = np.apply_along_axis(clr_normalize_column, 1, x)
    return normed_matrix

def adjust_adata(adata):
    obs = adata.obs.copy()
    uns = adata.uns.copy()
    obsm = adata.obsm.copy()
    obsp = adata.obsp.copy()
    # varm = adata.varm.copy() # Don't add the varm — new object will have different sets of genes, it just contains the PC loadings

    raw_mrna = get_raw_mrna(obs.index).copy()
    raw_mrna

    new_adata = raw_mrna.copy()

    sc.pp.normalize_total(new_adata, target_sum=1e6)
    sc.pp.log1p(new_adata)

    raw_adts = get_raw_adts(new_adata.obs_names).copy()

    adts = raw_adts.copy()
    new_adata.obs['adts_n_counts'] = adts.X.toarray().sum(axis=1)
    

    sc.pp.normalize_total(adts, target_sum=1e6)
    adts.X = clr_normalize(adts.X.toarray())

    new_adata = new_adata.T.concatenate(adts.T).T
    new_adata.var_names = [i[:-2] for i in new_adata.var_names]

    raw = raw_mrna.T.concatenate(raw_adts.T).T
    raw.var_names = [i[:-2] for i in raw.var_names]
    new_adata.raw = raw

    new_adata.obs = obs.copy()
    new_adata.uns = uns.copy()
    new_adata.obsm = obsm.copy()
    new_adata.obsp = obsp.copy()

    return new_adata
