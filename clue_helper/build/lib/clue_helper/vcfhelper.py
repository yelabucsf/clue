'''
Hi! This is vcfhelper module for working with VCFs

'''

import pandas as pd
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle as pkl
from tqdm.notebook import tqdm
import itertools as it
import anndata

import subprocess
import gzip
import sys
import os

def get_numheader(vcf_path):
    with gzip.open(vcf_path, 'rt') as file:
        i = 0
        for line in file:
            if line[0] == '#':
                i += 1
            else:
                break
    return i

def quantize(df, is_phased):
    if is_phased:
        quantize_geno_dict =     {
            '0|0': 0, '0|1': 1,
            '1|0': 1, '1|1': 2
        }
    else:
        quantize_geno_dict =     {
            '0/0': 0, '0/1': 1, '1/1': 2
        }

    return df.replace(quantize_geno_dict).astype(np.int8).copy()

default_dtypes = {
    'AF': float,
    'MAF': float,
    'R2': float,
    'ER2': float,
    'chr': str,
    'pos': float,
    'ref': str,
    'alt': str
}

def loadVCF(path, store_phase=False, dtypes=None):
    '''
    
    
    '''
    meta = dict()    
    meta['path'] = path

    # want to skip a certain number, assuming your VCF does not have a header > 1000 lines

    meta['num_header'] = get_numheader(path) - 1 # still want to keep the last line with sample info

    # read the file
    f = gzip.open(path,'rt')
    header = list()
    for i in range(meta['num_header']):
        header.append(f.readline())

    # create a list to build
    gts_list = list()
    info_list = list()
    uids = list()


    # add the single-line header
    colnames = f.readline().strip().split('\t')

    # for the rest of them, take the most important information only
    print("Reading in file.")
    for line in tqdm(f):
        line = line.strip().split('\t')

        uid = ":".join([line[0], line[1], line[3], line[4]])
        uids.append(uid)

        # TODO: option to store all the other data described in FORMAT besides GT
        gts = [i.split(":")[0] for i in line[9:]]
        gts_list.append(gts)

        infoline = [i.split('=') for i in line[7].split(";") if "=" in i]
        info_list.append(dict(zip([i[0] for i in infoline], [i[1] for i in infoline])))

    f.close()

    print("Converting to DataFrame... ", end='')
    gts_df = pd.DataFrame(gts_list, index=uids, columns=colnames[9:])
    gts_df.index.name='ID'
    meta['original_num_sites'] = gts_df.shape[0]
    print("Done.")

    # check phase
    head_num = 100 if gts_df.shape[0] > 100 else gts_df.shape[0]

    if gts_df.head(head_num).apply(lambda x: x.str.contains('|')).any().any():
        is_phased = True
        if store_phase:
            meta['phase'] = {
                0: np.where(gts_df.values == '1|0'),
                1: np.where(gts_df.values == '0|1')
            }
    elif gts_df.head(head_num).apply(lambda x: x.str.contains('/')).any().any():
        is_phased = False
    else:
        raise ValueError("Cannot read in genotype values.")

    meta['is_phased'] = is_phased

    # Currently, multi-allelic sites not supported
    # multi-allelic sites are marked by a comma in the ALT allele
    comma = gts_df.index.str.contains(',')

    if comma.sum() > 0:
        print("Warning: dropping %d multi-allelic site(s), adding to meta['dropped']." % comma.sum())
        dropsites = gts_df.index[comma]
        dropped_gts_df = gts_df.loc[dropsites]
        gts_df.drop(index=dropsites, inplace=True)

        dropped_info_list = list()
        for i in np.where(comma)[0]:
            dropped_info_list.append(info_list.pop(i))

        # keeping info_list as list causes problems
        # cannot implicitly convert non-str to str, so making it df
        dropped = {'gts_df': dropped_gts_df, 
                   'info_list': pd.DataFrame(dropped_info_list)} 
        
        
        meta['dropped'] = dropped

    meta['header'] = ''.join(header)

    print("Converting to AnnData... ", end='')
    gts_df = quantize(gts_df, is_phased)
    adata = anndata.AnnData(X=sparse.csr_matrix(gts_df.values), 
                            obs=pd.DataFrame(info_list, index=gts_df.index), 
                            var=pd.DataFrame(index=colnames[9:]))
    
    uids_expand = adata.obs.index.str.split(':', expand=True).to_frame()
    uids_expand.columns = ['chr', 'pos', 'ref', 'alt']
    uids_expand.index = adata.obs.index.copy()
    adata.obs = adata.obs.join(uids_expand)
    
    if not isinstance(dtypes, type(None)):
        assert isinstance(dtypes, dict)
        for dtype_k in dtypes:
            default_dtypes[dtype_k] = dtypes[dtype_k]
    for col in adata.obs.columns:
        try:
            adata.obs[col] = adata.obs[col].astype(default_dtypes[col])
        except KeyError:
            continue

    adata.uns['vcf_meta'] = meta
    print("Done.")
    
    return adata

def liftover_bed(bed_as_list, chain_abs_path, output_dir='tmp/', keep_files=False, output_name=None):
    '''
    Function for lifting over a list of lists of genomic regions (i.e. list([chrom, start, end])) from
    one genome to another, returning a list of lists. Wraps command line tool LiftoverBED. Output dir
    (default: output_dir=='tmp/') is where files will be written, will be created if does not exist 
    already, should have enough free disk space to write bed files. By default, keep_files==False, so 
    intermediate files will be deleted. Output name is required if keep_files==True.
    
    There is a known issue with the liftover tool where it fails with "Chain mapping error". I've figured
    out this can be avoided if the regions are of at least length 1 (so no identical start and end). A
    check is performed to make sure bed as list does not have identical start and end. 
    '''
    
    which_out = str(subprocess.run(['which', 'LiftoverBED'], stdout=subprocess.PIPE).stdout, encoding=sys.getdefaultencoding())
    if which_out.strip().split('/')[-1] != 'LiftoverBED':
        raise ValueError(
            '''
            LiftoverBED is not installed. Please install from 
            https://genome.sph.umich.edu/wiki/LiftOver, and add to shell PATH
            as `LiftoverBED`.
            '''
        )
    
    for i in bed_as_list:
        if i[1] == i[2]:
            raise ValueError('Known error of liftover tool. Avoid identical start and end loci.')
    
    if output_dir[-1] != '/':
        output_dir += '/'

    if keep_files == False:
        if type(output_name) != type(None):
            print('keep_files == False so output_name is ignored')
        rand_id = str(np.random.randint(low=1000, high=9999))
        output_name = rand_id + '_tmp'
    else:
        if type(output_name) != str:
            raise ValueError('You must set output_name if keep_files == True')
        elif output_name[-4:] == '.bed':
            output_name = output_name[:-4]

    try:
        os.mkdir(output_dir)
        if keep_files == False:
            delete_dir = True
        else:
            delete_dir = False
    except FileExistsError:
        delete_dir = False
    bed_as_list_name = output_name + '_pre_liftover'
    unlifted_name = output_name + '_unlifted'

    with open(output_dir + bed_as_list_name + '.bed', 'w') as file:
        for i in bed_as_list:
            file.write('\t'.join(i) + '\n')

    if os.path.isfile(output_dir + output_name + '.bed'):
        ans = input('Warning – output file %s exists. File will be overwritten. Continue anyway? (y/n)' % (output_dir + output_name + '.bed'))
        if ans == 'y':
            proceed = True
        elif ans == 'n':
            proceed = False
        else:
            raise ValueError('Answer with lowercase y or n')
    else:
        proceed = True

    if proceed:
        completed_process = subprocess.run(['LiftoverBED', 
                                            bed_as_list_name + '.bed', 
                                            chain_abs_path, 
                                            output_name + '.bed',  
                                            unlifted_name + '.bed'], cwd=output_dir)

        with open(output_dir + output_name + '.bed', 'r') as file:
            out_bed_as_list = [i.strip().split('\t') for i in file.readlines()]
        return out_bed_as_list
    else:
        if keep_files == False:
            for fname in [bed_as_list_name, output_name, unlifted_name]:
                os.remove(output_dir + fname + '.bed')
        if delete_dir == True:
            os.rmdir(output_dir)
        return

def convert_dtype(adata, cols=None, dtype=None, cat=None):
    '''
    Convert specified columns in the adata.obs to string.
    Useful when performing string operations on columns that
    are by default converted into dtype 'category' by common
    anndata operations (e.g. saving to h5ad, plotting).
    
    
    '''
    if not isinstance(cols, type(None)):
        if isinstance(cols, (list, tuple, pd.Series, pd.Index, np.ndarray)):
            if not isinstance(dtype, type):
                raise ValueError("Param `dtype` must be type.")
            cols = dict(it.product(cols, [dtype]))
        elif isinstance(cols, dict):
            pass
        else:
            raise ValueError("Param `cols` not understood. Pass list-like (with dtype) or dict.")
            
        for col in cols:
            adata.obs[col] = adata.obs[col].astype(cols[col])
    
    if not isinstance(cat, type(None)):
        if isinstance(cat, (list, tuple, pd.Series, pd.Index, np.ndarray)):
            for col in cat:
                adata.obs[col] = adata.obs[col].astype('category')
        else:
            raise ValueError("Param `cat` must be list-like.")
    return

    
    
def liftover(vcfadata, chain_abs_path, genomes, make_unique=True, **kwargs):

    uids_as_bed = [
        [
            i.split(':')[0], 
            i.split(':')[1], 
            str(int(i.split(':')[1]) + 1),  # this plus 1 was necessary for samtools depth
            i] 
        for i in vcfadata.obs_names]

    lifted = liftover_bed(uids_as_bed, chain_abs_path, **kwargs)

    vcfadata.obs.index = vcfadata.obs.index.rename(genomes[0] + '_ID')
    vcfadata.obs.rename(columns={
        'chr': genomes[0] + '_chr',
        'pos': genomes[0] + '_pos'}, inplace=True)
    
    
    lift_obs_join = pd.DataFrame(lifted).set_index(3)[[0, 1]]
    lift_obs_join.columns = [genomes[1] + '_chr', genomes[1] + '_pos']
    
    vcfadata.obs = vcfadata.obs.join(lift_obs_join)

    convert_dtype(vcfadata, cols=[genomes[1] + '_chr', genomes[1] + '_pos', 'ref', 'alt'], dtype=str)
    
    vcfadata.obs[genomes[1] + '_ID'] = vcfadata.obs[genomes[1] + '_chr'] + ':' + vcfadata.obs[genomes[1] + '_pos'] + ':' + vcfadata.obs['ref'] + ':' + vcfadata.obs['alt']
    
    convert_dtype(vcfadata, cat=[genomes[1] + '_chr', 'ref', 'alt'])
    convert_dtype(vcfadata, cols=[genomes[1] + '_pos'], dtype=float)
    
    if make_unique:
        # if not made unique, genomes[1] + '_ID' will be stored
        # as categorical if there are even two values the same
        # which makes the object unnecessary large and slow to load
        vcfadata.obs = vcfadata.obs.reset_index().set_index(genomes[1] + '_ID')
        vcfadata.obs_names_make_unique()
        vcfadata.obs = vcfadata.obs.reset_index().set_index(genomes[0] + '_ID')
        
    convert_dtype(vcfadata, cols=[genomes[1] + '_ID'], dtype=str)
    
    return

def adjust_chr(vcfadata, replace=None, add=None, regex=False):
    
    def adjust(vcfadata, g_str, r=None, a=None, regex=False):
        idstr = g_str + 'ID'
        chrstr = g_str + 'chr'
        
        convert_dtype(vcfadata, cols=[idstr, chrstr], dtype=str)
        
        if isinstance(a, str):
            vcfadata.obs[idstr] = a + vcfadata.obs[idstr]
            vcfadata.obs[chrstr] = a + vcfadata.obs[chrstr]
        if isinstance(r, tuple):
            vcfadata.obs[idstr] = vcfadata.obs[idstr].str.replace(*r, regex=regex)
            vcfadata.obs[chrstr] = vcfadata.obs[chrstr].str.replace(*r, regex=regex)
            
        convert_dtype(vcfadata, cat=[chrstr])
        
        return
    
    if not (isinstance(replace, type(None)) or isinstance(add, type(None))):
        raise ValueError("Can only specific either replace or add.")
    
    iname = vcfadata.obs.index.name
    vcfadata.obs.reset_index(inplace=True)
    try:
        if isinstance(add, type(None)):
            if isinstance(replace, dict):
                for g in replace:
                    adjust(vcfadata, g_str=g + '_', r=replace[g], regex=regex)
            else:
                adjust(vcfadata, g_str='', r=replace, regex=regex)
        else: # isinstance(regex, type(None)) == True
            if isinstance(add, dict):
                for g in add:
                    adjust(vcfadata, g_str=g + '_', a=add[g])
            else:
                adjust(vcfadata, g_str='', a=add)
    except Exception as e:
        # catch any issues and return the object as it was
        vcfadata.obs.set_index(iname, inplace=True)
        raise e
    vcfadata.obs.set_index(iname, inplace=True)
    
    return      

def make_refalt(name_split):
    return {
        0.0: name_split[-2] + '/' + name_split[-2],
        1.0: name_split[-2] + '/' + name_split[-1],
        2.0: name_split[-1] + '/' + name_split[-1],
    }
