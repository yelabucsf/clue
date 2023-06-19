'''
Hi! This is the sctools module for general functions aiding in the analysis of single cell data.
'''
import scanpy as sc
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from tqdm.notebook import tqdm
from clue_helper import utils

import warnings
import itertools as it
import json
import pickle as pkl
import os
import sys

def relabel_clusts(adata, key='leiden'):
    '''
    Relabel the values in `key` as ordered categories numbering from 0 to _n_.
    
    `adata`: annotated data matrix
    `key`: name of column in `adata.obs` with the clusters
    
    returns: None, modifies in-place
    ''' 
    try:
        adata.obs[key].cat
    except AttributeError:
        adata.obs[key] = adata.obs[key].astype('category')
        
    cats = adata.obs[key].cat.categories
    new_cats = [str(i) for i in range(len(cats))]
    adata.obs[key] = adata.obs[key].map(dict(zip(cats, new_cats)))
    adata.obs[key] = adata.obs[key].astype('category')
    try:
        del(adata.uns[key + '_colors']) # so new ("correct") colors will be added upon plotting
    except KeyError:
        pass
    return

def subcluster(adata, **kwargs):
    '''
    Subcluster using `restrict_to`, then run `relabel_clusts` to relabel the clusters.
    
    `adata`: annotated data matrix
    `kwargs`: passed to `sc.tl.leiden()`, must include `key_added`
    
    returns: None, modifies in-place
    '''
    sc.tl.leiden(adata, **kwargs)
    relabel_clusts(adata, key=kwargs['key_added'])
    return

def regroup(adata, groupings, key='leiden'):
    '''
    Regroup clusters in `key` according to `groupings`.
    
    `adata`: annotated data matrix
    `groupings`: nested list of lists, or dict; if list of lists, sublists are how
                 clusters in `key` should be grouped; if dict, dict values are list 
                 of clusters that will be converted into dict key; does not require 
                 all groups to be listed
    `key`: name of column in `adata.obs` with the clusters
    
    returns: None, modifies in-place
    '''
    if isinstance(groupings, list):
        grouped_clusts = [i for j in groupings for i in j]
        numclusts = np.unique(adata.obs[key].values)
        for i in np.setdiff1d(numclusts, grouped_clusts):
            groupings.append([i])
            ctdict = dict()
        for i in range(len(groupings)):
            ctdict['ct%s' % str(i)] = groupings[i]

        adata.obs['celltype'] = adata.obs[key]
        for ct in ctdict:
            for clust in ctdict[ct]:
                adata.obs['celltype'].replace(str(clust), ct, regex=True, inplace=True)
        adata.obs[key] = [i.strip('ct') for i in adata.obs['celltype']]
        relabel_clusts(adata, key=key)
    elif isinstance(groupings, dict):
        inv_groupings = {vi: k  for k, v in groupings.items() for vi in v}
        adata.obs[key] = adata.obs[key].replace(inv_groupings).astype('category')
    print('New number of clusters: ' + str(len(adata.obs[key].cat.categories)))
    return

def freq_table(adata, cols):
    '''
    Make a frequency table of two different columns in the .obs of the adata
    object. Returns a df with the number of observations per combination of
    values from the two columns.
    
    `adata`: annotated data matrix
    `cols`: 2-tuple of the two columns of adata.obs to count
    
    returns: pd.DataFrame of frequencies
    '''
    c1, c2 = cols
    return adata.obs.groupby([c1, c2]).size().rename('count').reset_index(c2).pivot(columns=c2)
    

def lR_to_l(adata, mapper={'leiden_R': 'leiden'}):
    '''
    Map a current column name to a new column name. By default,
    maps `leiden_R` to `leiden`, typically run after using 
    `sc.tl.leiden(restrict_to=)`.
    
    `adata`: annotated data matrix
    
    returns: None, modifies in-place
    '''
    for current_col_name in mapper:
        new_col_name = mapper[current_col_name]
        current_col = adata.obs[current_col_name].copy()
        adata.obs.drop(columns=current_col_name)
        adata.obs[new_col_name] = current_col
    return

def subcluster_mapper(adata, sub_adatas):
    '''
    Map subclusters from `sub_adata` onto the original `adata` object. Only works with
    the `leiden` column.
    
    `adata`: annotated data matrix
    `sub_adatas`: dict of `adata` objects isolating a single cluster that was further
                  subclustered.
                  
    returns: `adata`, annotated data matrix
    '''
    clust_col = adata.obs['leiden'].astype(float)
    for k in sub_adatas:
        sub_adata = sub_adatas[k]
        clust_col.loc[sub_adata.obs_names] += sub_adata.obs['leiden'].astype(float)/1000 # there shouldn't be more than 1000 subclusters, right?
    unique_vals = sorted(clust_col.unique())
    new_clusts = list(map(str,range(len(unique_vals))))
    mapper = dict(zip(unique_vals, new_clusts))
    clust_col = clust_col.map(mapper).astype('category').cat.reorder_categories(new_clusts)
    adata.obs['leiden'] = clust_col.copy()
    return adata

def prop_col_chart(adata, group, x='leiden', norm=True, prop=True, 
                   ax=None, return_df=False, label=False, label_thresh=0.02, label_size=10):
    '''
    Plots a column chart using `adata.obs` showing the proportional coincidence of two covariates.
    
    `adata`: annotated data matrix
    `group`: name of column in `adata.obs` whose proportion will be plotted along the y axis
    `x`: name of column in `adata.obs` to be plotted along the x axis. Default: `leiden`.
    `norm`: whether or not to normalize the proportion of `group` in `x` by the total number of 
            `group` across the dataset. Useful when there are very few observations for a `group`
    `prop`: whether or not to plot the proportion or absolute number of `group` in `x`
    `ax`: matplotlib.Axes object on which to plot the data
    `return_df`: whether or not to return the pd.DataFrame grouped by (`group`, `x`), `df_gb`
    `label`: whether or not to label the data on the plot
    `label_thresh`: a threshold proportion below which the data should not be labeled. Useful
                    to avoid overlapping label text on the plot
    `label_size`: size of the text labels on the data
    
    
    returns: `ax` or (`ax`, `df_gb`), matplotlib.Axes object and/or pd.DataFrame(), 
             depending on `return_df`.
    '''
    
    df = adata.obs[[group, x]].copy()
    
    try: # if it's something that can be coerced into a number, probably want to display in numerical order
        df[group] = df[group].astype(int)
    except ValueError: # can't be coerced, oh well
        pass
    
    try: # if it's something that can be coerced into a number, probably want to display in numerical order
        df[x] = df[x].astype(int)
    except ValueError: # can't be coerced, oh well
        pass
    
    df_gb = df.reset_index(drop=False).groupby([group, x]).count().reset_index(group).pivot(columns=group)
    df_gb.columns = df_gb.columns.droplevel(0)
    df_gb.fillna(0, inplace=True)
    if norm:
        vcounts = df[group].value_counts()
        for col in df_gb:
            df_gb[col] = df_gb[col]/vcounts.loc[col]
    if prop:
        df_gb = df_gb.div(df_gb.sum(1), axis=0)
    
    df_gb.columns = df_gb.columns.astype(str) # sometimes they're coerced into numerical which causes problems
    df_gb.index = df_gb.index.astype(str) # sometimes they're coerced into numerical which causes problems
    
    if isinstance(ax, type(None)):
        fig, ax = plt.subplots(figsize=(8,5))
    lastpos = [0]*len(df_gb.index)
    
    try:
        for g, c in zip(adata.obs[group].cat.categories, adata.uns[group + '_colors']):
            ax.bar(df_gb.index, df_gb[g].values, label=str(g), bottom=lastpos, color=c)
            if label:
                for idx, y, lp in zip(df_gb.index, df_gb[g].values, lastpos):
                    label = '{:03.1f}'.format((y)*100) if prop else str(y)
                    if y < label_thresh:
                        continue
                    else:
                        ax.text(idx, lp + y/2, s=label, ha='center', va='center', size=label_size)
            lastpos = df_gb[g].values + lastpos
    except KeyError:
        for g in adata.obs[group].cat.categories:
            ax.bar(df_gb.index, df_gb[g].values, label=str(g), bottom=lastpos)
            if label:
                for idx, y, lp in zip(df_gb.index, df_gb[g].values, lastpos):
                    label = '{:03.1f}'.format((y)*100) if prop else str(y)
                    if y < label_thresh:
                        continue
                    else:
                        ax.text(idx, lp + y/2, s=label, ha='center', va='center', size=label_size)
            lastpos = df_gb[g].values + lastpos
    if prop:
        if norm:
            ax.set_ylabel('Normalized proportion')
        else:
            ax.set_ylabel('Proportion')
    else:
        if norm:
            ax.set_ylabel('Normalized absolute number')
        else:
            ax.set_ylabel('Absolute number')
    ax.grid(False)
    ax.set_xticks(df_gb.index)
    
    if not return_df:
        return ax
    else:
        return ax, df_gb

def get_dge(adata):
    '''
    Restructure the rank_genes_groups output of sc.tl.rank_genes_groups()
    so that the results are in a single DataFrame.
    
    `adata`: annotated data matrix
    
    returns: `deg_data`, a pd.DataFrame with the DE data
    '''
    deg_data = pd.DataFrame()
    groups = adata.uns['rank_genes_groups']['scores'].dtype.names
    n_genes = adata.uns['rank_genes_groups']['scores'].shape[0]
    for i in ['scores', 'names', 'logfoldchanges', 'pvals', 'pvals_adj']:
        deg_data[i] = np.array(adata.uns['rank_genes_groups'][i].tolist()).flatten()
    deg_data['group'] = list(groups)*n_genes
    deg_data = pd.concat([deg_data[deg_data['group'] == group].sort_values(by='scores', ascending=False) for group in groups], axis=0)
    return deg_data

def add_lowde(adata, key='leiden', groups=None):
    '''
    Calculate the percentage of cells of certain group(s) expressing at least one
    count of each genes. In conjunction with low_de_compare(), useful for identifying 
    sets of genes that are lowly expressed but highly specific to certain groups.
    
    `adata`: annotated data matrix
    `key`: name of column in `adata.obs` with the groups to compare
    `groups`: optional, a list-like of length 2, with list-likes of values in 
              `adata.obs[key]`, for analyzing groups of cells jointly. In addition to
              each value in `adata.obs[key]`, percentage expression results will contain 
              information for 'g1' and 'g2', according to the order provided.
              
    returns: None, `adata` modified in-place with new key 'lowde' in `adata.uns`
    '''
    lowde = dict()
    groups_provided = not isinstance(groups, type(None))
    if groups_provided:
        assert isinstance(groups, list), 'groups must be a list of 2 lists'
        assert isinstance(groups[0], list) and isinstance(groups[1], list), 'groups must be a list of 2 lists'
        if not all([i in adata.obs[key].cat.categories for i in [j for k in groups for j in k]]):
            print('Warning: groups not in %s. Trying converted to str.' % key)
            assert all([i in adata.obs[key].cat.categories for i in [str(j) for k in groups for j in k]]), 'groups not in %s' % key
            groups = [list(map(str, g)) for g in groups]
            
    
    labels = adata.obs[key].dtype.categories.tolist()
    gs = [[i] for i in labels]
    if groups_provided:
        g1, g2 = groups
        labels.append('g1')
        labels.append('g2')
        gs.append(g1)
        gs.append(g2)
    
    
    try:
        X = adata.raw.X.copy()
        cols = adata.raw.var_names.values
    except AttributeError:
        print('Warning: no .raw attribute. Using .X directly — .X should NOT be scaled.')
        X = adata.X.copy() # this should NOT be scaled
        cols = adata.var_names.values
        
    lowde['df'] = pd.DataFrame(0, index=labels, columns=cols, dtype=np.float16)
    for g, label in zip(gs, labels):
        clustbool = (adata.obs[key].isin(g)).values
        clustX = X[clustbool]
        try:
            clustX = clustX.tocsc()
        except AttributeError:
            from scipy.sparse import csc_matrix
            clustX = csc_matrix(clustX)
        lowde['df'].loc[label] = clustX.getnnz(axis=0)/clustX.shape[0]
    if groups_provided:
        lowde['groups'] = dict(zip(['g1', 'g2'], [g1, g2]))
    adata.uns['lowde'] = lowde
    return

def low_de_compare(adata, compare, p, p_of='any'):
    '''
    Uses `adata.uns['lowde']` and calculates a ratio (compare[0]/compare[1]) of 
    percentage of cells expressing each feature. Filter features by `p`, a minimum
    percentage of cells expressing, before calculating the ratios. Param `p_of` dictates
    which groups should be used for the minimum percentage filter `p`.
    
    `adata`: annotated data matrix
    `compare`: which groups to calculate the ratio, of list-like of length 2
    `p`: minimum percentage filter for features before calculating ratio
    `p_of`: which group(s) to calculate the minimum percentage filter for. Can be one of:
            'any': the minimum percentage must be met by at least one group in 
                   `adata.uns['lowde'].index`
            val(s): a single value (or list-like of multiple values) in
                    `adata.uns['lowde'].index` for which the minimum percentage filter will
                    apply. If multiple values, all group of cells must meet the minimum
                    percentage. 
    
    returns: pd.Series of index=features and values of ratios
    
    Note: If `add_lowde` was provided with param `groups`, groups `g1` and `g2` will be
          in `adata.uns['lowde'].index`, and these groups will be considered if `p_of` is
          'any'. Note that if `p_of` is multiple values, *each* group will have to meet
          the minimum percentage, whereas if a group `g1` is a group made up of those same
          values (as provided to add_lowde), the feature might only be expressed in one group
          while still meeting the minimum percentage. This difference is useful in cases 
          where a group of lowly-expressed features is specific to multiple groups, *and*
          another set of lowly-expressed features is specific to each group. Each group 
          should express the minimum percentage of the shared specific features.
    '''
    
    assert len(compare) == 2, 'Can only compare 2 groups, len(compare) must == 2'
    
    df = adata.uns['lowde']['df']
    if p_of == 'any':
        sub_df = df.iloc[:,df.apply(lambda x: np.any(x > p), axis=0, raw=True).values]
    else:
        assert len(np.intersect1d(np.ravel([p_of]), df.index)) > 0 , "p_of not in adata.uns['lowde']['df'].index"
        if not isinstance(p_of, (tuple, list, np.ndarray)):
            sub_df = df.iloc[:,(df.loc[p_of] > p).values]
        elif isinstance(p_of, (tuple, list, np.ndarray)):
            sub_df = df.iloc[:, pd.concat([df.loc[g] > p for g in p_of], axis=1).all(1).values]
        else:
            raise ValueError("Param `p_of` not understood. Should be value in, or list-like subset of adata.uns['lowde']['df'].index.")

    return (sub_df.loc[compare[0]]/sub_df.loc[compare[1]]).sort_values(ascending=False)

def rank_plot(dge=None, xytxt=None, size=10, ax=None):
    '''
    Similar to sc.pl.rank_genes_groups(), but swaps axes and displays gene names
    horizontally. Must provide one of dge or xytxt.
    
    `dge`: dge object created by get_dge
    `xytxt`: 3-tuple of lists of x values, y values, and strings of text 
             to display at each point
    `size`: fontsize of the text annotations on the plot
    `ax`: matplotlib axes object to plot on, will create if not provided
    
    returns: `ax`, matplotlib axes object
    '''
    ### THIS DOESN'T WORK FOR MULTIPLE PLOTS (i.e. sc.tl.rank_genes_groups(adata, groups=['group1', 'group2']))
    if isinstance(ax, type(None)):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        
    if isinstance(dge, type(None)):
        x, y, txts = xytxt
    else:
        n_features = dge.shape[0]//len(dge['group'].unique())
        y = range(n_features)[::-1]
        x = dge['scores'].values
        txts = dge['names'].values
        
    ax.scatter(x, y, s=0)
    ax.barh(np.array(y) + 0.25, x, facecolor='lightgray', edgecolor='k')
    for i, txt in enumerate(txts):
        ax.annotate(txt, (x[i]*1.02, y[i]), rotation=0, size=size)
    ax.set_yticklabels([])
    ax.spines['right'].set_visible(False)
    
    return ax
    
    

def grouped_rank(adata, return_dge=True, key='leiden', method='t-test_overestim_var', 
                 n_genes=20, bg='gray', size=5, figsize=(5, 5), overwrite=False):
    '''
    Run sc.tl.rank_genes_groups() using the groupings added by `add_lowde()`. The function
    highlights the groups in UMAP space and returns a single ranked gene plot (with gene 
    names displayed horizontally).
    
    `adata`: annotated data matrix
    `return_dge`: boolean, whether or not to return the dge pd.DataFrame
    `key`: name of column in `adata.obs` with the clusters
    `method`: the method to use for sc.tl.rank_genes_groups()
    `n_genes`: the number of genes to show and optionally return
    `bg`: background color, accepted by matplotlib.ax.set_facecolor()
    `size`: dot size of the UMAPs
    `figsize`: figsize of the UMAPs
    `overwrite`: whether or not to overwrite the `adata.uns['rank_genes_groups']` with 
                 what is run here, or keep previously-run results
     
    returns: None or dge DataFrame if return_dge == True
    '''
    groups = list(adata.uns['lowde']['groups'].values())
    
    try:
        adata.uns['rank_genes_groups']
    except KeyError: # doesn't exist
        overwrite = True

    grouped_clusts = [i for j in groups for i in j]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for clusts, title, ax in zip(groups, ['0','1'], axes):
        adata.obs['val'] = adata.obs[key].isin(clusts).astype(float)
        ax.set_facecolor(bg)
        sc.pl.umap(adata,color='val', ax=ax, size=size, show=False, return_fig=False, title=title, legend_loc=None)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
    adata.obs.drop(columns='val', inplace=True)

    adata.obs['rank_compare'] = pd.concat(list(map(adata.obs[key].isin, groups)), axis=1).astype(str).sum(1).replace({
        'TrueFalse': '0',
        'FalseTrue': '1',
        'FalseFalse': 'N/A'
    }).astype('category')
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if not overwrite:
        original_rgg = adata.uns['rank_genes_groups'].copy()

    sc.settings.verbosity = 0
    sc.tl.rank_genes_groups(adata, groupby='rank_compare', method=method, n_genes=n_genes, groups=['0'], reference='1', use_raw=True)
    dge = get_dge(adata)

    if not overwrite:
        adata.uns['rank_genes_groups'] = original_rgg

    ax = rank_plot(dge=dge, ax=ax)
    
    ax.set_title('0 vs 1')
    sc.settings.verbosity = 4
    adata.obs.drop(columns='rank_compare', inplace=True)
    if return_dge:
        return dge
    else:
        return

# def grouper(n, iterable):
#     '''
#     Iteration tools recipe, from itertools.
#     '''
#     iterable = iter(iterable)
#     while True:
#         chunk = tuple(it.islice(iterable, n))
#         if not chunk:
#             return
#         yield chunk

def plot_features(adata, features, bg='gray', ncols=5, 
                  figsize_scale=1.5, show=True, aspect_ratio=(3.5, 3.1), 
                  lims=None, **kwargs):
    '''
    
    `adata`: annotated data matrix
    `features`: is a list-like or dict of features in `.var_names`; if dict,
                keys will be printed before plotting, values are list-like of features
    `bg`: background color, accepted by matplotlib.ax.set_facecolor()
    `ncols`: the number of columns to plot
    `figsize_scale`: scale of the figsize; default (1) yields fig of size (20, 3.75)
    `show`: whether or not to show the figure(s) as they're created, only when
            `features` is dict; otherwise they all show at once
    `aspect_ratio`: aspect ratio of each individual plot, adjust to make the plot square
    `kwargs`: passed to `sc.pl.umap()`
    
    
    returns: figaxes, a list or dict of fig and axes objects
    '''
    
    
    def adjust_lims(ax, lims):
        if isinstance(lims, list):
            ax.set_xlim(lims[0])
            ax.set_ylim(lims[1])
        else:
            try:
                ax.set_xlim(lims['x'])
            except KeyError:
                pass
            try:
                ax.set_ylim(lims['y'])
            except KeyError:
                pass
        return ax
    
    
    def plotter(fs, **kwargs):
        nrows = int(np.ceil(len(fs)/5))
        std_figsize = np.array([ncols*aspect_ratio[0], nrows*aspect_ratio[1]])
        fig, axes = plt.subplots(nrows, ncols, figsize=tuple(std_figsize*figsize_scale))
        for color, ax in zip(fs, np.ravel(axes)):
            ax.set_facecolor(bg)
            ax = sc.pl.umap(adata, color=color, ax=ax, show=False, return_fig=False, **kwargs)
            if not isinstance(lims, type(None)):
                ax = adjust_lims(ax, lims)
        num_ax_remove = ncols - (len(fs) % ncols)
        if (num_ax_remove > 0) and (num_ax_remove < ncols):
            for ax in np.ravel(axes)[-num_ax_remove:]:
                ax.set_visible(False)
        if show:
            plt.show()
        return (fig, axes)
    
    if isinstance(features, dict):
        figsaxs = dict()
        for feature_set_name in features:
            fs = features[feature_set_name]
            print('########### ' + feature_set_name + ' ###########')
            figsaxs[feature_set_name] = plotter(fs, **kwargs)
            print('')
            print('')
    elif isinstance(features, (list, tuple, np.ndarray, pd.Series)):
        if isinstance(features, pd.Series):
            features = pd.Series.values
        figsaxs = plotter(features, **kwargs)
        print('')
        print('')
    else:
        raise ValueError('Features must be a dict or list-like')
    return figsaxs

def highlight_clust(adata, key, clusts, bg='lightgray', bg_cells='gray', 
                    other_label='other', plot_kwargs={}):
    '''
    Highlight cluster(s) in UMAP space with background color.
    
    `adata`: annotated data matrix
    `key`: name of column in `adata.obs` with the clusters
    `clusts`: member(s) of `adata.obs[key]` to highlight
    `bg`: background color, accepted by matplotlib.ax.set_facecolor()
    `bg_cells`: color of non-highlighted cells
    `figsize`: size of resulting figure
    `other_label`: key added to adata.obs used internally, but can be provided
                   with str if default 'other' is already in use in adata.obs
    
    returns: `ax`, matplotlib Axes object
    '''
    
    if isinstance(clusts, str):
        label = key + ':' + clusts
        clusts = [clusts]
    elif isinstance(clusts, list):
        clusts_len = len(clusts)
        if clusts_len <= 3:
            label = key + ':' + '&'.join(clusts)
        else:
            label = key + ':' + str(clusts_len) + '_clusts'
        
    other_clusts = np.setdiff1d(np.unique(adata.obs[key]), clusts)
    adata.obs[label] = adata.obs[key].copy()
    adata.obs[label].replace(dict(it.product(other_clusts, [other_label])), inplace=True)
    adata.obs[label] = adata.obs[label].astype('category')
    
    key_cdict = dict(zip(adata.obs[key].cat.categories, adata.uns[key + '_colors']))
    label_cdict = dict()
    for clust in adata.obs[label].cat.categories:
        if clust == other_label:
            label_cdict[clust] = bg_cells
        else:
            label_cdict[clust] = key_cdict[clust]
    
    adata.uns[label + '_colors'] = list(label_cdict.values())
    
    fig, axes = plot_features(adata, features=[label], bg=bg, **plot_kwargs)
    for ax in np.ravel(axes):
        ax.set_facecolor(bg)
    adata.obs.drop(columns=[label], inplace=True)
    
    return ax

# def highlight_feature(adata, feature, bg='gray', use_raw=False, 
#                       figsize=(8,8), size=20, ax=None, **kwargs):
#     '''
#     Highlight a single feature in UMAP space with background color.
    
#     `adata`: annotated data matrix
#     `feature`: feature name in `.var_names`
#     `bg`: background color, accepted by matplotlib.ax.set_facecolor()
#     `use_raw`: use the raw attribute of `adata`
#     `figsize`: size of resulting figure
#     `size`: dot_size of UMAP scatter
#     `kwargs`: passed to `sc.pl.umap()`
    
#     returns: `ax`, matplotlib Axes object
#     '''
    
#     if isinstance(ax, type(None)):
#         fig, ax = plt.subplots(1, 1, figsize=figsize)
        
#     ax.set_facecolor(bg)
#     sc.pl.umap(adata, color=feature, ax=ax, return_fig=False, use_raw=use_raw, 
#                show=False, size=size, **kwargs)
#     return ax

def check_var(adata, sw=None, ew=None, cont=None):
    '''
    Check feature names in adata using startswith, endswith, and
    contains filters. If multiple filters provided, returned results
    satisfy all filters.
    
    `adata`: annotated data matrix
    `sw`: str supplied to startswith when checking feature names
    `ew`: str supplied to endswith when checking feature names
    `cont`: str supplied to contains when checking feature names
    
    return: filtered adata.var_names, of type pd.Index
    '''
    return_vars = list()
    if not isinstance(sw, type(None)):
        return_vars.append(adata.var_names.str.startswith(sw))
    if not isinstance(ew, type(None)):
        return_vars.append(adata.var_names.str.endswith(ew))
    if not isinstance(cont, type(None)):
        return_vars.append(adata.var_names.str.contains(cont))
    return adata.var_names[np.stack(return_vars).all(0)]
    
def excise_umap(adata, 
                cbs=None, xyr=None, pick=False, show=False, 
                pick_params=None,
                cbs_params=None,
                show_params=None
               ):
    '''
    Extract the xlim and ylim of an adata's UMAP that would
    zoom in on a particular area of the UMAP. You can specify: 

    (1) specific cell barcodes (`cbs`), with optional pad/crop 
        (`buffer`) and `offsets`.
        
    (2) an area of the map defined by an x, y center and radius, 
        using the units of the "pick plot" (see below). The returned 
        xlim and ylim will be x +/- r, y +/- r.
        
    If pick==True, a UMAP plot with gridlines will display. The gridlines
    are in the units defined by `pick_params`. You may observe it and choose
    an x, y, and radius then run the function again with `xyr`.
        
    
    `cbs`: a list of cell barcodes to zoom in on, ignored if 
           `pick==True`
    `xyr`: the x, y, and r radius using the units of the deafult or
           user provided "pick plot"
    `pick`: boolean, if you would like to visualize the "pick plot"
    `show`: show what the resulting UMAP will look like with the new
            xlim and ylim
    `pick_params`: dict, may include:
                       'pick_num' (number of ticks on resulting UMAP, 
                       default=20)
    `cbs_params`: dict, may include:
                      `buffer`, a float as a percentage of full UMAP space
                      that pads (positive) or crops (negative) frame by 
                      constant amount on all sides; 
                      `offsets`, a 2-tuple of offset frame by a percentage 
                      of full umap space, which can be positive (shift frame
                      up/right) or negative (down/left)
    `show_params`: dict, params provided to nr.plot_features; if none provided,
                   the default feature that will be plotted is adata.obs.columns[0]
    
    returns: umap_xlim, umap_ylim; the range of xlim and ylim that centers view 
             of UMAP space on the provided area, without distortion
    '''
    params_sum = sum([not isinstance(cbs, type(None)), not isinstance(xyr, type(None)), pick])
    assert params_sum == 1, "Must provide one of `xyr`, `cbs`, or pick=True."
    
    default_pick_params = {'pick_num': 20}
    default_cbs_params = {'buffer': 0.0, 'offsets': (0.0, 0.0)}
    default_show_params = {'features': adata.obs.columns[:1].tolist()}
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax = sc.pl.umap(adata, ax=ax, show=False, return_fig=False)
    (full_umap_xmin, full_umap_xmax), (full_umap_ymin, full_umap_ymax) = ax.get_xlim(), ax.get_ylim()
    full_umap_xrange, full_umap_yrange = full_umap_xmax - full_umap_xmin, full_umap_ymax - full_umap_ymin
    
    if not isinstance(cbs, type(None)):
        plt.close()
        cbs_params = utils.replace_none(cbs_params, default_cbs_params)
        buffer = cbs_params['buffer']
        offsets = cbs_params['offsets']
    
        # get the full scale umap xlim and ylim


        # get the specific X and Y values, and mins/maxes, of the cells
        X, Y = adata[cbs, :].obsm['X_umap'].T.toarray()
        xmax, xmin, ymax, ymin = X.max(), X.min(), Y.max(), Y.min()

        # define the lims, ranges, and midpoints of the x and y data
        data_xlim, data_ylim = np.array([xmin, xmax]), np.array([ymin, ymax])
        data_xrange, data_yrange = xmax - xmin, ymax - ymin
        data_xmid, data_ymid = data_xlim.mean(), data_ylim.mean()

        # adjust frame to a square b/c UMAPs are squares, use the range of the axis that's bigger
        # need to scale by the full umap lims, otherwise clusters look skewed/stretched
        max_percent_of_full = max([data_xrange/full_umap_xrange, data_yrange/full_umap_yrange])
        new_xrange, new_yrange = full_umap_xrange*max_percent_of_full, full_umap_yrange*max_percent_of_full

        # prepare the buffer and offsets
        offx, offy = -offsets[0]*full_umap_xrange, -offsets[1]*full_umap_yrange
        buffx, buffy = buffer*full_umap_xrange, buffer*full_umap_yrange

        # compute the new xlim and ylim
        new_xlim = (data_xmid - new_xrange/2 + offx - buffx, data_xmid + new_xrange/2 + offx + buffx) 
        new_ylim = (data_ymid - new_yrange/2 + offy - buffy, data_ymid + new_yrange/2 + offy + buffy)
        
        if show:
            show_params = utils.replace_none(show_params, default_show_params)
            plot_features(adata, **show_params, lims=[new_xlim, new_ylim])
        return new_xlim, new_ylim
    else:
        try:
            pick_params = utils.replace_none(pick_params, default_pick_params)
            pick_num = pick_params['pick_num']
            xtix = np.linspace(full_umap_xmin, full_umap_xmax, num=pick_num + 1, endpoint=True)
            ytix = np.linspace(full_umap_ymin, full_umap_ymax, num=pick_num + 1, endpoint=True)
            unit_x, unit_y = xtix[1] - xtix[0], ytix[1] - ytix[0]
        except:
            e = sys.exc_info()[0]
            plt.close()
            raise e
        
        if not isinstance(xyr, type(None)):
            plt.close()
            X, Y, radius = xyr
            new_xlim = (unit_x*(X - radius)+full_umap_xmin, unit_x*(X + radius)+full_umap_xmin)
            new_ylim = (unit_y*(Y - radius)+full_umap_ymin, unit_y*(Y + radius)+full_umap_ymin)
            
            if show:
                show_params = utils.replace_none(show_params, default_show_params)
                plot_features(adata, **show_params, lims=[new_xlim, new_ylim])
            return new_xlim, new_ylim
        elif pick == True:
            ax.set_xticks(xtix)
            ax.set_xticklabels(range(pick_num + 1))
            ax.set_yticks(ytix)
            ax.set_yticklabels(range(pick_num + 1))
            ax.grid(zorder=3)
            plt.show()
            return
        else:
            raise ValueError('Param `pick` not understood.')
            
def matrixplot(adata, features, key='leiden', func=np.mean, restrict_to=None, 
               standardize=None, use_raw=False, order=None, figsize=(8, 3), fsscale=1,
               plot_params=None, landscape=True, cluster=None):
    '''
    if standardize provided, annotations will still be in original units
    
    '''

    try:
        if not isinstance(plot_params['norm'], type(None)) and not isinstance(standardize, type(None)):
            print("Warning: `plot_params['norm'] ignored when `standardize` provided.")
    except:
        pass
    
    def run_func(adatadf):
        try:
            dfplot = adatadf.groupby(key)[features].apply(func, axis=0)
        except TypeError:
            dfplot = adatadf.groupby(key).apply(func, axis=0)
            
        if isinstance(dfplot, pd.Series) and len(features) > 1:
            new_dfplot = pd.DataFrame(dfplot.to_dict()).T
            new_dfplot.columns = features
            dfplot = new_dfplot
            
        assert len(dfplot.shape) == 2, "Param `func` is incompatible."
        
        return dfplot
                
    
    if not isinstance(restrict_to, type(None)):
        adata = adata[utils.subsetdict(adata.obs, restrict_to).index]
    if not use_raw:
        adatadf = adata[:, features].to_df()
    else:
        adatadf = adata[:, features].raw.to_df()
    adatadf = adatadf.join(adata[:, features].obs[[key]])
    dfplot = run_func(adatadf)
    
    if not isinstance(order, type(None)):
        dfplot = dfplot.loc[order, :]

    
    norm = mpl.colors.TwoSlopeNorm(vmin=0, vcenter=0.2, vmax=4)
    default_plot_params = dict()
    default_plot_params.update(annot=dfplot, fmt='0.2f', norm=norm, 
                               square=True, linewidth=1, linecolor='black',
                               cmap='viridis', cbar=None, annot_kws={'size': 8})
    plot_params = utils.replace_none(plot_params, default_plot_params)
    
    if not landscape: # this does not work as expected
        dfplot = dfplot.T
        default_plot_params['annot'] = dfplot
    
    if not isinstance(standardize, type(None)):
        plot_params['norm'] = None
        dfplot_std = utils.standard_scale(dfplot, standardize)
        dfplot_std = dfplot_std.fillna(0) # arises when the values are all 0, division by zero
        
        dfplot_vals = dfplot_std
    else:
        dfplot_vals = dfplot
    
    if not isinstance(cluster, type(None)):
        del(plot_params['square'])
        if fsscale != 1:
            print("Warning: param `fsscale` ignored with cluster != None. Set `figsize` manually.")
        if cluster == True: # any string will satisfy `if`, but not equate to True, so need to check True
            cg = sns.clustermap(dfplot_vals, figsize=figsize, **plot_params)
        elif cluster == 'key':
            if landscape:
                cg = sns.clustermap(dfplot_vals, figsize=figsize, 
                                    row_cluster=True, col_cluster=False, **plot_params)
                cg.ax_col_dendrogram.set_visible(False)
                cg.ax_cbar.set_visible(False)
            else:
                cg = sns.clustermap(dfplot_vals, figsize=figsize, 
                                    row_cluster=False, col_cluster=True, **plot_params)
                cg.ax_row_dendrogram.set_visible(False)
                cg.ax_cbar.set_visible(False)
        elif cluster == 'features':
            if landscape:
                cg = sns.clustermap(dfplot_vals, figsize=figsize, 
                                    row_cluster=False, col_cluster=True, **plot_params)
                cg.ax_row_dendrogram.set_visible(False)
                cg.ax_cbar.set_visible(False)
            else:
                cg = sns.clustermap(dfplot_vals, figsize=figsize, 
                                    row_cluster=True, col_cluster=False, **plot_params)
                cg.ax_col_dendrogram.set_visible(False)
                cg.ax_cbar.set_visible(False)
        else:
            raise ValueError("Param `cluster` not understood. Either 'key' or 'features'.")
        return dfplot, cg
    else:
        fig, ax = plt.subplots(1, 1, figsize=tuple(np.array(figsize)*fsscale))
        ax = sns.heatmap(dfplot_vals, ax=ax, **plot_params)
        if landscape:
            ax.tick_params(bottom=False, labelbottom=False, top=True, labeltop=True)
            
            # something is wrong with these
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='left')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=90, ha='right')
        else:
            ax.tick_params(bottom=False, labelbottom=False, top=True, labeltop=True)
    return dfplot, ax
    
