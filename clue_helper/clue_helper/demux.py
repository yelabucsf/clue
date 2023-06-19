'''
Hi! This is a helper module for analyzing outputs of Demuxlet

'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from copy import copy
import scanpy as sc

class MuxOut:
    '''
    Class storing the `.samples` output of freemuxlet. Reports metadata and has useful functions for \
    extracting out droplets, plotting, and visualizing distributions. 
    
    path (str): path to the demultiplexing output (demuxlet/popscle: `.best` file; vireo: `donors.tsv`)
    label (str): if you'd like to add a specific label (stored in .label) you can add it here
    
    useful_cols (boolean): this will store a `.data` attribute which is relied on by several methods. \
    If you'd like to forgo creating this attribute (save memory) you can set it to False. When this is \
    False, metadata is not stored. You can run the .useful_cols() and .store_meta() method 
    after importing if desired.

    TODO: 
    one of the demuxlet outputs is reporting individuals as "IGTB###,IGTB###,0.50" in BEST column, need to recognize this

    Todo: DOCSTRING!


    
    '''
    global stdcolumns
    stdcolumns = ['NUM.SNPS','NUM.READS','DROPLET.TYPE','BEST.GUESS']
    
    def __init__(self, path, label=None, useful_cols=True):
        

        def infer_muxtype(columns):
            if columns[0] == 'RD.TOTL':
                muxtype = 'OD'
                selftype = 'old_demux'
                usefulcols = ['N.SNP','BEST','RD.TOTL']
            elif columns[0] == 'NUM.SNPS':
                muxtype = 'PD'
                selftype = 'popscle_demux'
                usefulcols = stdcolumns
            elif columns[0] == 'INT_ID':
                muxtype = 'PF'
                selftype = 'popscle_freemux'
                usefulcols = stdcolumns
            elif columns[0] == 'donor_id':
                muxtype = 'VO'
                selftype = 'vireo'
                # even though the raw does not have these columns, the raw columns will
                # be adjusted when calling useful_cols() to create the data attribute
                # and will match the ones shown here
                usefulcols = ['NUM.SNPS','BEST.GUESS','DROPLET.TYPE']
            else:
                raise Exception('Columns were not recognizable, could not infer muxtype.')
            return muxtype, selftype, usefulcols

        self.path = path
        self.label = label
        try:
            raw = pd.read_csv(path,sep='\t').set_index('BARCODE') # read in files
        except KeyError:
            raw = pd.read_csv(path,sep='\t').set_index('cell') # for vireo
        except KeyError:
            raise Exception("Muxlet output type not understood.")
        
        self.raw = raw
        self.muxtype, self.type, self.usefulcols = infer_muxtype(self.raw.columns)
        
        self.useful_cols_run = False
        self.store_meta_run = False
        if useful_cols == True:
            self.useful_cols()
            self.store_meta()

    def useful_cols(self):
        
        # make old demuxlet match popscle
        if self.muxtype == 'VO':
            # vireo stores doublets in separate column, merging it into one "best" column
            # for a few doublets, doesnt label but instead reports two names separated by comma, very similar to freemuxlet
            # confirmed in the summary.tsv output by vireo that they're supposed to be interepretted as doublets
            data = self.raw.copy()
            # replace messed up doublets, but if there's a comma in the name (will vireo even allow that?) this will be messed up
            data['donor_id'].replace(r'.*,.*','doublet',regex=True, inplace=True)

            droptype = data['donor_id'].copy() # manipulate a separate Series

            # regex replacement is super fast!
            droptype.replace(r'doublet','DBL',regex=True, inplace=True)
            droptype.replace(r'unassigned','AMB',regex=True, inplace=True)
            droptype.replace(r'^(?!DBL|AMB).*$','SNG',regex=True, inplace=True)

            # I want donor_id column to have comma-separated names still, not just "doublet" as it is natively
            data['donor_id'].where((droptype == 'SNG') |
                                  (droptype == 'AMB'), data['best_doublet'],inplace=True)

            data.rename_axis(index='BARCODE', inplace=True) # keep it consistent with demux
            data.drop('best_doublet',axis=1,inplace=True) # rid us of now useless column
            data['DROPLET.TYPE'] = droptype # keep it consistent with demux
            
            data = data[['n_vars','donor_id','DROPLET.TYPE']].copy() # subsetting columns here
            data.columns = ['NUM.SNPS','BEST.GUESS','DROPLET.TYPE'] # renaming them for consistency
            
            self.data = data # no need to subset self.usefulcols because I already did it in the line above
        else:
            self.data = self.raw[self.usefulcols].copy()
            
            if self.muxtype == 'OD':
                # all of the following to avoid Pandas SettingwithCopy warning
                newdata = self.data.copy()
                newcols = self.data['BEST'].str.split('-', n = 1, expand = True)
                newdata[['DROPLET.TYPE','BEST.GUESS']] = newcols

                # now replace and put in correct order
                del(newdata['BEST'])
                newdata.columns = stdcolumns
                self.data = newdata
            
        self.useful_cols_run = True
        
    def reassign_thresh(self, threshold=-2):
        '''
        Function to re-assign DBL and SNG based on DIFF.LLK.SNG.DBL field.
        NOTE: cell will not be assigned to AMB.
        ANOTHER NOTE: Need to deconvolute BEST.GUESS
        TODO: make compatible with other inputs. So far works for muxtype=PD/PF
        :param threshold: difference in LLK between assignment this cell as SNG or DBL. Default = -2
        :return: void. But replaces self.data with new assignments.
        '''

        assert self.muxtype in ["PD", "PF"], "reassign_thresh Error: only available for " \
                                            "popscle demuxlet and popscle freemuxlet outputs"

        diff_llk_list = self.raw["DIFF.LLK.SNG.DBL"].tolist()
        new_assign = ["SNG" if x > threshold else "DBL" for x in diff_llk_list]
        self.raw['DROPLET.TYPE'] = new_assign

    def make_inds(self):

        # check integrity of singlets, sometimes DROPLET.TYPE is called SNG but BEST.GUESS has two individuals
        sngs = self.sng()
        sngindex = sngs.index

        # pull out the individuals, note this is only the set of individuals found in singlets
        # i.e. if certain inviduals are only found in doublets, they will not show here
        if self.muxtype == 'PF':
            inds = [list(set(i)) for i in sngs['BEST.GUESS'].str.split(',')]
        else:
            self.inds = list(set(sngs['BEST.GUESS'].tolist()))
        if self.muxtype == 'PF':
            if set([len(i) for i in inds]) != {1}:
                warnings.warn('Some singlets have two different individuals. Label them as doublets with clean_sngs().')
            else:
                # change the BEST.GUESS column to just report one number for singlets
                self.data.loc[sngindex,'BEST.GUESS'] = [i[0] for i in inds]
                num_inds = max([int(i) for i in set(self.sng()['BEST.GUESS'])]) + 1
                self.inds = [str(i) for i in range(num_inds)]   

    def store_meta(self):
        # store some meta data
        self.num_drops = len(self.data)
        self.num_sng = len(self.sng())
        self.num_dbl = len(self.dbl())
        self.num_amb = len(self.amb())
        
        self.sngrate = self.num_sng/(self.num_drops)
        self.dblrate = self.num_dbl/(self.num_drops)
        self.ambrate = self.num_amb/(self.num_drops)
        
        self.sngrate_nonamb = self.num_sng/(self.num_drops - self.num_amb)
        self.dblrate_nonamb = self.num_dbl/(self.num_drops - self.num_amb)

        self.make_inds()
        
        self.store_meta_run = True

    def sng(self):
        if not self.useful_cols_run:
            print('Creating the .data attribute in order to run this function.')
            self.useful_cols()
        return self.data[self.data['DROPLET.TYPE'].str.contains("SNG") == True]

    def dbl(self):
        if not self.useful_cols_run:
            print('Creating the .data attribute in order to run this function.')
            self.useful_cols()
        return self.data[self.data['DROPLET.TYPE'].str.contains("DBL") == True]

    def amb(self):
        if not self.useful_cols_run:
            print('Creating the .data attribute in order to run this function.')
            self.useful_cols()
        return self.data[self.data['DROPLET.TYPE'].str.contains("AMB") == True]

    def sngdbl(self):
        if not self.useful_cols_run:
            print('Creating the .data attribute in order to run this function.')
            self.useful_cols()
        return pd.concat([self.sng(), self.dbl()])

    def sngamb(self):
        if not self.useful_cols_run:
            print('Creating the .data attribute in order to run this function.')
            self.useful_cols()
        return pd.concat([self.sng(), self.amb()])

    def dblamb(self):
        if not self.useful_cols_run:
            print('Creating the .data attribute in order to run this function.')
            self.useful_cols()
        return pd.concat([self.dbl(), self.amb()])

    def snp_plot(self, ax=None, listonly = 0):
        if not self.useful_cols_run:
            print('Creating the .data attribute in order to run this function.')
            self.useful_cols()
        snpsorted = copy(self.data['NUM.SNPS'].sort_values(ascending=False))
        if listonly > 0 :
            return snpsorted[:listonly]
        if ax == None:
            plt.figure(figsize=(5,5))
            plt.plot(snpsorted.tolist())
            plt.yscale('log')
            plt.xscale('log')
            plt.ylabel('SNPs')
            plt.xlabel('Cells')
            plt.grid(which='minor',alpha=0.3)
            plt.grid(which='major',alpha=0.7)
            if self.label != None:
                plt.title(self.label)
        else:
            ax.plot(snpsorted.tolist())
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_ylabel('SNPs')
            ax.set_xlabel('Cells')
            ax.grid(which='minor',alpha=0.3)
            ax.grid(which='major',alpha=0.7)
            if self.label != None:
                ax.set_title(self.label)
            return ax
            

#     def clean_sngs(self):
#         change self.data directly to remove them

    def ind_hist(self, ax=None, figsize=(5,5),use_trail_chars=None):
        if not self.store_meta_run:
            print('Creating metadata attributes in order to run this function.')
            self.store_meta()

        samplist = list(self.sng()['BEST.GUESS'])
        counts = list()
        for ind in self.inds:
            counts.append(samplist.count(ind))
        if ax == None:
            plt.figure(figsize=figsize)
            plt.bar(self.inds,counts)
            if use_trail_chars != None:
                plotinds = [i[-use_trail_chars:] for i in self.inds]
            else:
                plotinds = self.inds
            locs, labels = plt.xticks()
            plt.xticks(ticks=locs, labels=plotinds, rotation=45)
            plt.xlabel('Individuals')
            plt.ylabel('Counts')
            if self.label != None:
                plt.title(self.label)
        else:
            ax.bar(self.inds,counts)
            if use_trail_chars != None:
                plotinds = [i[-use_trail_chars:] for i in self.inds]
            else:
                plotinds = self.inds
            ax.set_xticklabels(labels=plotinds, rotation=45)
            ax.set_xlabel('Individuals')
            ax.set_ylabel('Counts')
            if self.label != None:
                ax.set_title(self.label)
            return ax
    
    def llk_plot(self, ax=None, listonly = 0, figsize=(5,5)):
        if not self.useful_cols_run:
            print('Creating the .data attribute in order to run this function.')
            self.useful_cols()
            
        if self.type == 'vireo':
            print('Log liklihood plot not yet supported for vireo outputs.')
            return
        raw = self.raw
        if self.type == 'old_demux':
            dropdiff = raw['LLK12'] - raw['SNG.LLK1']
        elif self.type == 'popscle_demux':
            # I noticed after observing the popscle demuxlet outputs that the columns start getting messed up
            # after the 'NEXT.GUESS' column, the headers don't match the data type and the last column
            # (which I'd like to use) doesn't contain any data (NaN). I realized they're actually just shifted one
            # (this may also be related to how the table also doesn't contain the INT_ID column) 
            # So I'm going to take the data in the second to last column, 'DBL.BEST.LLK' because this actually
            # represents the 'DIFF.SNG.DBL.LLK' data. This might have been run with the development version of
            # popscle, so I should change this back when its resolved.
            warnings.warn('Using DBL.BEST.LLK column for %s because columns may be shifted. See source for comment.' % self.label)
            dropdiff = raw['DBL.BEST.LLK']
        else:
            dropdiff = raw['DIFF.LLK.SNG.DBL']
        
        minimum = min(dropdiff)
        
        subdf = pd.concat([self.data['DROPLET.TYPE'],dropdiff],axis=1)
        subdf.columns = ['DROPLET.TYPE','DROP.DIFF']
        
        subdf = subdf.sort_values('DROP.DIFF')
        if listonly > 0:
            return subdf.iloc[:listonly,:]
        mylist = list(zip(subdf['DROP.DIFF'],subdf['DROPLET.TYPE'],range(len(raw))))
        if ax == None:
            plt.figure(figsize=figsize)
        for i in zip(['DBL','AMB','SNG'],['g','k','r']):
            drop = i[0]
            color = i[1]
            x = np.array([j[2] for j in mylist if drop in j[1]])
            y = np.array([j[0] for j in mylist if drop in j[1]])
            if drop == 'AMB':
                # subtrating minimum just for display purposes
                if ax == None:
                    plt.plot(x,y - minimum,label=drop,color=color,linewidth=10,alpha=0.25)
                else:
                    ax.plot(x,y - minimum,label=drop,color=color,linewidth=10,alpha=0.25)
            else:
                # subtrating minimum just for display purposes
                if ax == None:
                    plt.plot(x,y - minimum,label=drop,color=color,linewidth=10,alpha=0.25)
                else:
                    ax.plot(x,y - minimum,label=drop,color=color,linewidth=10,alpha=0.25)
            if ax == None:
                plt.legend()
            else:
                ax.legend()
        if ax == None:
            plt.yscale('log')
            plt.ylabel('Difference in log liklihoods')
            plt.xlabel('Cell number')
            if self.label != None:
                plt.title(self.label)
        else:
            ax.set_yscale('log')
            ax.set_ylabel('Difference in log liklihoods')
            ax.set_xlabel('Cell number')
            if self.label != None:
                ax.set_title(self.label)
            return ax

    def filter_snps(self, value, bcs=False):
        '''
        Filter the associated data. If bcs is false, value is a threshold by which to filter based on number
        of SNPs per barcode. If bcs is True, value is the number of cells to keep with the highest SNPs.
        Meta-data (e.g. singlet rate, number of drops) updates with filtered data.
        '''
        if not self.useful_cols_run:
            print('Creating the .data attribute in order to run this function.')
            self.useful_cols()
        if bcs == False:
            self.data = self.data[self.data['NUM.SNPS'] >= value]
        elif bcs == True:
            keep_bcs = self.data.sort_values(by='NUM.SNPS',ascending=False).index[:value]
            self.data = self.data.loc[keep_bcs]
        else:
            assert ValueError, 'bcs not set correctly'

        self.store_meta()

def heatmap(mux1, mux2, ax=None, droptype='SNG', sum_filt=None, sz=7, return_data=None, map_thresh=None):
    '''
    Function that takes two MuxOut outputs and makes a heatmap that colors boxes based on
    how many times the droplet was called as a given pair of individuals between the two outputs.
    Can take in singlets and doublets.

    mux1, mux 2
    MuxOut objects

    droptype: 'SNG', 'DBL', or 'AMB'
    only handles SNG right now, should expand to others

    sum_filt: [min_row, min_col]
    filter for any row or column (individual) in the heatmap that has less than a total number of barcodes called
    must provide both if provided at all; use 0 if do not want to filter for one dimension

    return_data: 'df' or 'map'
    in addition to returning the figure, also return the underlying dataframe ('df') or a list of mappings
    between individuals, given a map_thresh threshold for overlapping barcodes. map_thresh must be set.

    map_thresh: int >= 0
    a threshold (inclusive) of overlapping barcodes to consider to return the mapping between two MuxOut objects.
    if return_data != 'map', this is ignored.
    '''
    
    ### TODO: A matplotlib figure is being returned even when not requested, fix it
    if droptype == 'SNG':
        calls = mux1.sng()[['BEST.GUESS']].join(mux2.sng()[['BEST.GUESS']],how='inner',lsuffix='-1',rsuffix='-2')
        calls.columns = [1,2]

    assert calls.shape[0] > 0, "No overlapping barcodes"

    call_names1 = np.unique(calls[1].values)
    call_names2 = np.unique(calls[2].values)

    # TODO: warn about less than 100 overlapping barcodes
    heatmapdf = pd.DataFrame(np.zeros((len(call_names1),len(call_names2)),
                                      dtype=int),
                             columns=sorted(call_names2),
                             index=sorted(call_names1))


    for i in range(len(calls)):
        id1 = calls.iloc[i][2]
        id2 = calls.iloc[i][1]
        heatmapdf[id1][id2] += 1

    if sum_filt != None:
        assert len(sum_filt) == 2, "Provide minimum sum filters for both rows and columns"

        # should test these
        heatmapdf = heatmapdf[heatmapdf.sum(axis=1) > sum_filt[0]]
        heatmapdf = heatmapdf.loc[:,heatmapdf.sum(axis=0) > sum_filt[1]]

    # TODO: return a list of mappings based on a ratio of highest to second-highest, inputtable parameter
    if ax == None:
        plt.figure(figsize=(sz,sz))
    if return_data == None:
        return sns.heatmap(heatmapdf,annot=True,annot_kws={'fontsize':10},fmt="d",ax=ax)
    elif return_data == 'df':
        return heatmapdf
    elif return_data == 'map':
        assert map_thresh != None, "If requesting mappings, map_thresh must be provided"
        maps = list()
        for row in heatmapdf.index:
            for col in heatmapdf.columns:
                val = heatmapdf.loc[row,col]
                if val >= map_thresh:
                    maps.append([(row,col), val])
        return maps

def ann_merge(adata, mux, covars=None):
    '''
    Merge a mux output (and optionally a dataframe with individual covariates) into the `obs` 
    of the adata provided. Parameter `mux` can either be a MuxOut class (to keep all cells) or the 
    DataFrame output of a MuxOut droptype method (e.g. mux.sng(), mux.sngdbl(), etc.). Parameter `covars` 
    must be a DataFrame with an index containing the names of indivduals as found in the BEST.GUESS 
    column of the MuxOut object, and covariates as columns. This function requires Scanpy.
    '''
    myadata = adata.copy()
    if mux.__class__ == MuxOut:
        new_obs = pd.concat([myadata.obs, mux.data], 
                        axis=1, join='inner')
    elif mux.__class__ == pd.DataFrame:
        new_obs = pd.concat([myadata.obs, mux], 
                        axis=1, join='inner')
    myadata = myadata[new_obs.index,:].copy()
    myadata.obs = new_obs
    if type(covars) != type(None):
        for covar in covars.columns:
            try:
                myadata.obs[covar] = list(map(lambda x: covars.loc[x,covar],
                                              myadata.obs['BEST.GUESS'])) # this adds the covars to the H5
            except KeyError:
                print('Value in BEST.GUESS from MuxOut object not in covars. Please check that the ' + \
                'unique BEST.GUESS values are identical to the covars index.')
                return
    return myadata

    
    
    
    
    
