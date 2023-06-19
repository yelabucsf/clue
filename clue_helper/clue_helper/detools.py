'''
The detools module for functions aiding in analysis of DE results.
'''

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from tqdm.notebook import tqdm
from copy import deepcopy 
from clue_helper.utils import enhanced_roll
from anndata.utils import make_index_unique

import requests

# KMeans and Hierarchical Clustering Convenience Functions
def get_dists(df, ks, metric='euclidean'):
    '''
    Get distortions with varying _k_ for _k_-means clustering. 
    [Source for finding optimal k](https://pythonprogramminglanguage.com/kmeans-elbow-method/).

    `df`: pd.DataFrame on which to perform k_means clustering, with observations
          as columns and variables as as rows
    `ks`: 1D ndarray of values for k to test; typically produced with np.arange() 
    `metric`: distance metric
    
    returns: dists, of type list

    '''
    dists = list()
    for k in tqdm(ks):
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(df)
        dists.append(sum(np.min(cdist(df, kmeanModel.cluster_centers_, metric), axis=1)) / df.shape[0])
    return dists
    
    
def plot_dists(dists, ks, slopes_plot='bar', slopes_bar_params=None, slopes_line_params=None):
    '''
    Plot the distortions and slope between each k.
    [Source for finding optimal k](https://pythonprogramminglanguage.com/kmeans-elbow-method/).

    `dists`: list of dists produced with get_dists()
    `ks`: 1D ndarray of values for k to test; typically produced with np.arange()
    `slopes_plot`: type of plot for slopes, either bar or line with moving window     
    `slopes_bar_params`: params for slopes_bar plot; default is 
                         {'width': 2, 'color': 'k', 'alpha': 0.5}
    `slopes_line_params`: params for slopes_bar plot; default is 
                          {'window': 5, 'color': 'k'}

    returns: ax, list of 2 matplotlib Axes objects
    '''
    assert slopes_plot in ['bar', 'line']
    assert len(dists) == len(ks)
    
    _slopes_bar_params = {'width': 2, 'color': 'k', 'alpha': 0.5}
    if not isinstance(slopes_bar_params, type(None)):
        assert isinstance(slopes_bar_params, dict)
        for key in slopes_bar_params:
            _slopes_bar_params[key] = slopes_bar_params[key]

    _slopes_line_params = {'window': 5, 'color': 'k'}
    if not isinstance(slopes_line_params, type(None)):
        assert isinstance(slopes_line_params, dict)
        for key in slopes_line_params:
            _slopes_line_params[key] = slopes_line_params[key]
            
    rise = [dists[i] - dists[i - 1] for i in range(1, len(ks))]
    run = [ks[i] - ks[i - 1] for i in range(1, len(ks))]
    slopes = np.array([-i/j for i, j in zip(rise, run)])

    # Plot the elbow
    fig, ax = plt.subplots(2, 1, figsize=(6,6))
    ax[0].plot(ks, dists - min(dists) + 1, 'bx-')
    ax[0].set_ylabel('Distortion')
    if slopes_plot == 'bar':
        ax[1].bar(ks[1:], slopes, **_slopes_bar_params)
        ax[1].set_ylabel('Slopes')
    else:
        s = pd.Series(slopes)
        s = enhanced_roll(s, _slopes_line_params['window'])
        del(_slopes_line_params['window'])
        ax[1].plot(ks[1:], s.values, **_slopes_line_params)
        

    return ax

def distortion_plot(df, ks, metric='euclidean'):
    '''
    Calculate and plot the dists with varying _k_ for _k_-means clustering.
    Figure produces two plots: (1) dists with _k_, and (2) slopes between _k_.
    
    `df`: pd.DataFrame on which to perform k_means clustering, with observations
          as columns and variables as as rows
    `ks`: 1D ndarray of values for k to test; typically produced with np.arange() 
    `metric`: distance metric
    
    returns: tuple of (dists, ax), of type (list, list); 
             values of list of type (float, matplotlib Axes object)
    
    '''
    
    dists = get_dists(df, ks, metric='euclidean')
    ax = plot_dists(dists, ks)
    
    return dists, ax   

def get_ordering(Z, labels, return_idxs=False):
    '''
    Get the ordering of a linkage matrix. 
    [Source](https://stackoverflow.com/questions/12572436/calculate-ordering-of-dendrogram-leaves)
    
    `Z`: linkage matrix from scipy.cluster.hierarchy.linkage()
    `labels`: np.ndarray, pd.Index, or pd.Series with labels; typically
              the index of the clustered axis
    `return_idxs`: return the indices that reorder the labels instead of the
                   re-ordered labels
                   
    returns: re-ordered `labels` or indices of re-ordered labels, depending 
             on `return_idxs`
             
    '''
    
    n = len(Z) + 1
    cache = dict()
    for k in range(len(Z)):
        c1, c2 = int(Z[k][0]), int(Z[k][1])
        c1 = [c1] if c1 < n else cache.pop(c1)
        c2 = [c2] if c2 < n else cache.pop(c2)
        cache[n+k] = c1 + c2
    ordering = cache[2*len(Z)]
    if return_idxs == False:
        return np.array(labels)[ordering]
    elif return_idxs == True:
        return ordering
    else:
        raise ValueError

# Functional Enrichment Using ToppFun API
headers = {'Content-Type': 'application/json'}

def get_mapper(genes, return_unfound=True):
    '''
    
    
    '''
    
    genes_str = ','.join(['"' + str(i) + '"' for i in genes])
    
    lookup_url = 'https://toppgene.cchmc.org/API/lookup'
    lookup_data = '{"Symbols":[' + genes_str + ']}'
    
    response = requests.post(lookup_url, headers=headers, data=lookup_data).json()
    mapper = dict(zip([i['Submitted'] for i in response['Genes']], [i['Entrez'] for i in response['Genes']]))
    if return_unfound:
        unfound = list(set(genes).difference(set(mapper.keys())))
        return mapper, unfound
    else:
        return mapper
    
class toppfun:
    '''
    Functional enrichment output from ToppFun API available at 
    https://toppgene.cchmc.org/API/enrich.
    
    Currently accepts list of HUGO-formatted gene symbols.
    Instantiate with a list of genes. 
    
    Params:
    `genes`: list of gene symbols in HUGO (human)
    `mapper`: dict mapping gene symbols to Entrez IDs, created with
              get_mapper(); if not supplied, will be run;
    `unfound`: list of unfound gene symbols; if not supplied  
    
    Methods:
    enrich() — call the API to run functional enrichment
    
    '''
    

        
    def __init__(self, genes_all):
        
        if np.unique(genes_all).shape[0] != len(genes_all):
            print('Genes not unique; making unique.')
            # get unique without sorting:
            indexes = np.unique(genes_all, return_index=True)[1]
            genes_all = [genes_all[index] for index in sorted(indexes)]
            
        self.genes_all = genes_all
        self.genes_query = None
        self.genes_enrich = None
        self.mapper, self.unfound = get_mapper(genes_all, return_unfound=True)
        
        return
                
    
    def enrich(self, genes_query, verbose=False):
        '''
        Run a functional enrichment of a list of genes using the toppgene
        
        Attributes added:
        
        `p`: percent of genes in annotation
        `results`: pd.DataFrame with enrichment results

        returns: None, modifies in-place
                  attribute
        '''
        
        self.genes_query = genes_query

        ids = list()
        for i in self.genes_query:
            try:
                ids.append(self.mapper[i])
            except KeyError:
                continue

        self.p_query = len(ids)/len(self.genes_query)
        
        if verbose:
            print('Only %d out of %d (%s) supplied genes converted.' % (len(ids), len(self.genes), str((self.p_query*100))[:5] + '%'))
        
        enrich_url = 'https://toppgene.cchmc.org/API/enrich'
        headers = {'Content-Type': 'application/json'}
        enrich_data = '{"Genes":[' + ','.join([str(i) for i in ids]) + ']}'
        
        response = requests.post(url=enrich_url, 
                                 headers=headers, 
                                 data=enrich_data)
        
        results = pd.DataFrame(response.json()['Annotations'])
        results['-log10FDR'] = -np.log10(results['QValueFDRBH'])
        results.set_index('ID', inplace=True)
        dupes = results.index.duplicated().sum()
        if dupes > 0:
            if verbose:
                print('Ontology IDs duplicated: %d. Making unique.' % dupes)
            results.index = make_index_unique(results.index)
            
        self.genes_enrich = dict(zip(results.index, [[j['Symbol'] for j in i] for i in results['Genes']]))
        results.drop(columns=['Genes'], inplace=True)
        self.results = results

        return 
    
    def copy(self):
        '''
        Returns a copy.deepcopy() of the toppfun object.
        
        '''
        return deepcopy(self)
    
    def res_filter(self, query=None, cols=['Category', 'Name', '-log10FDR'], genes=True):
        '''
        '''
        
        res_filt = self.results.copy()
        
        if not isinstance(query, type(None)):
            res_filt.query(query, inplace=True)
            
        res_filt = res_filt.loc[:, cols]
        
        if genes:
            res_filt['Genes'] = res_filt.index.map(self.genes_enrich)

        return res_filt
