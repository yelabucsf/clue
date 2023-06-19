'''
Utils for clue_helper. Little functions used throughout the package multiple times.
'''

import subprocess
import sys
import shlex
import pandas as pd
import numpy as np
import subprocess

def shellSort(infile, outfile, tempdir, verSort=False):
    '''
    Function to perform bash sorting
    :param infile: Str. Input file
    :param outfile: Str. Output file
    :param verSort: Boolean. Use version sorting. Default = TRUE
    :return: Void.
    '''
    v = "V" if verSort else ""
    with open(outfile, 'w') as sorted_file:
        subprocess.run(['sort', '-T', tempdir, '-k1,1%s' %v, '-k2,2n', infile],
                       stdout=sorted_file)


def overlapFilter(infile, outfile, verbose=False):
    with open(infile, 'r') as peak_file:
        with open(outfile, 'w') as filt_peak_path:
            count = 0
            curr_site = [0] * 5
            for line in peak_file:
                count += 1
                site = line.strip().split('\t')
                # test if peaks overlap
                # NOTE first line will be 0 - will fix it later insted of doing thousands of 'if' tests
                if int(site[1]) - int(curr_site[1]) > 501:
                    filt_peak_path.write("\t".join(str(x) for x in curr_site) + '\n')
                    curr_site = site.copy()

                # if peaks overlap
                else:
                    # use most significant peak
                    if float(site[-1]) > float(curr_site[-1]):
                        curr_site = site.copy()
                    else:
                        continue

                # deal with the last line
                if site == ['']:
                    filt_peak_path.write("\t".join(str(x) for x in curr_site) + '\n')
                    break

                if verbose and count % 50000 == 0:
                    print('%d peaks read.' % count)
                    sys.stdout.flush()
    # remove first line of output file
    command = "sed -i 1d"
    args = shlex.split(command)
    args.append(outfile)
    p = subprocess.Popen(args)

def enhanced_roll(s, window, func=np.mean):
    '''
    Performs a rolling window calculation but fills in the NAs that 
    inevitably form on the end with rolling calculations using smaller
    and smaller windows. First value is the same, second value is 
    _func_ of first 2, 3rd first 3, until w, when all calculations become 
    _func_ of rolling w.
    
    `s`: pd.Series with values
    `window`: size of the moving window, see pd.Series.rolling()
    `func`: function used to calculate rolling statistic, will be used in
    
    
    returns: s, pd.Series with rolling statistic
    
    '''
    if window > 1: 
        return s.rolling(window=window).apply(func).fillna(enhanced_roll(s, window-1))
    else:
        return s
    
def make_logspace(start, stop, num, endpoint=True, dtype=None, axis=0):
    '''
    Wrapper for np.logspace but input unlogged start and stop. Because
    unlogged data is provided, base parameter is irrelevant and therefore
    not inputtable.
    
    start : array_like
        ``log(start)`` is the starting value of the sequence.
    stop : array_like
        ``log(stop)`` is the final value of the sequence, unless `endpoint`
        is False.  In that case, ``num + 1`` values are spaced over the
        interval in log-space, of which all but the last (a sequence of
        length `num`) are returned.
    num : integer, optional
        Number of samples to generate.  Default is 50.
    endpoint : boolean, optional
        If true, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.
    axis : int, optional
        The axis in the result to store the samples.  Relevant only if start
        or stop are array-like.  By default (0), the samples will be along a
        new axis inserted at the beginning. Use -1 to get an axis at the end.


    returns: `num` samples, equally spaced on a log scale.
    '''

    return np.logspace(start=np.log10(start), stop=np.log10(stop), num=num, endpoint=endpoint, dtype=dtype, axis=axis)

def adj_light(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def subsetdict(df, sdict):
    '''
    Subset a pd.DataFrame with a dictionary where keys are column names and 
    values are value (or multiple values = list()) to subset on.
    
    `df`: pd.DataFrame to subset
    `sdict`: dictionary of {col1: val1, col2: val2} or, if multiple values,
             {col1: [val1, val2], col2: val3}.
    
    `returns`: subset_df, subsetted pd.DataFrame
    
    '''
    subsetter_list = [df[i].isin([j]) if not isinstance(j, list) else df[i].isin(j) for i, j in sdict.items()]
    subsetter = pd.concat(subsetter_list, axis=1).all(1)
    return df.loc[subsetter, :]


def id_axes(ax, lim=None, tix=None):
    '''
    Will create perfectly identical x and y axes, with exactly the same
    range, tick locations and tick labels, while still showing all data. 
    Defaults to use the locations and labels of the axis with a large number 
    of tick locations, but can be provided with `tix`. The axes limits `lim` 
    may also be provided, otherwise will be computed using the min and max limit
    of either axis. Labels will be at *all* tick locations.
    
    `ax`: matplotlib.axes object
    `lim`: 2-tuple of (low_lim, hi_lim), axis limits to use for both axes
    `tix`: list-like of tick locations accepted by ax.set_*ticks()
    
    '''
    
    if isinstance(lim, type(None)):
        lim = min(min(ax.get_xlim()), min(ax.get_ylim())), max(max(ax.get_xlim()), max(ax.get_ylim()))
    
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    
    if isinstance(tix, type(None)):
        tix = ax.get_xticks() if len(ax.get_xticks()) > len(ax.get_yticks()) else ax.get_yticks()
    ax.set_xticks(ticks=tix)
    ax.set_yticks(ticks=tix)
    
    tix_labels = np.where((tix % 1).astype(bool), tix.astype(str), tix.astype(int).astype(str))
    ax.set_xticklabels(labels=tix_labels)
    ax.set_yticklabels(labels=tix_labels)
    return ax

def norm_between(arr, start, spread):
    '''
    Normalize an array from start with spread.
    
    `arr`: np.ndarray to normalize
    `start`: start of the normalized values
    `spread`: the spread of the normalized values
    
    returns: normalized np.ndarray
    '''
    return (arr - arr.min())/arr.ptp() * spread + start

def func_mid(arr, midrange=90, func=np.mean):
    '''
    Compute a function on the middle `midrange` percent of a numpy array.
    
    `arr`: 1-D np.ndarray
    `midrange`: the middle percentage of values for which `func` will be calculated
    `func`: the function to apply to the values, default np.mean, takes in a 1-D 
            np.ndarray

    `returns`: the computed value
    '''
    offset = (100 - midrange)/2
    lims = np.percentile(arr, q=[offset, 100 - offset])
    x_sub = x[((np.abs(arr) > lims[0]) & (np.abs(arr) < lims[1]))]
    
    if arr_sub.shape[0] > 0:
        return func(arr_sub)
    else:
        return 0
    
    
def read_memlog(path, parse_date=True, time_from='min'):
    '''
    Read the memlog file that is created with the mem_stats.sh
    shell script:
    
    ```
    #! /bin/bash
    set -e
    printf "Time\t\tMemory\t\tDisk\t\tCPU\n"
    end=$((SECONDS+360000)) # this is 100 hours
    while [ $SECONDS -lt $end ]; do
    TIME=$(date)
    MEMORY=$(free -m | awk 'NR==2{printf "\t\t%.2f%%\t\t", $3*100/$2 }')
    DISK=$(df -h | awk '$NF=="/"{printf "%s\t\t", $5}')
    CPU=$(top -bn1 | grep load | awk '{printf "%.2f%%\t\t\n", $(NF-2)}')
    echo "$TIME$MEMORY$DISK$CPU"
    sleep 5
    done
    ```
    
    Run in a screen with sh mem_stats.sh >> mem_log.txt. Then run
    the program for which you'd like to monitor memory usage.
    
    `path`: path to mem_log.txt
    `parse_date`: parse the date using dateparser, requires to be
                  installed (https://dateparser.readthedocs.io)
    `time_from`: with dateparser, format the time as "time from start",
                 default 'sec', will accept 'min' & 'hr'
    
    returns: `mem`, a pd.DataFrame with memory usage
    
    '''
    
    
    
    
    def map_dates(mem, time_from):
        mem.index = mem.index.map(dateparser.parse)
        
        if time_from:
            deltas = [0]
            for i in mem.index[1:]:
                deltas.append((i - mem.index[0]).total_seconds())
            deltas = np.array(deltas)
            if time_from == 'sec' or time_from == True:
                pass
            elif time_from == 'min':
                deltas = deltas/60
            elif time_from == 'hr':
                deltas = deltas/3600
            else:
                raise ValueError('Param `time_from` not understood.')
            mem.index = deltas
            
        return mem
    
    mem = pd.read_csv(path, sep='\t')
    mem.drop(columns=[i for i in mem.columns if 'Unnamed' in i], inplace=True)
    mem = mem.droplevel(level=1).rename_axis('Time', axis=0)
    mem = mem.iloc[:, :-1].copy()
    mem.columns=['Memory', 'Disk', 'CPU']
    for i in mem.columns:
        mem[i] = mem[i].str.rstrip('%').astype(float)

    if parse_date:
        try:
            mem = map_dates(mem, time_from)
        except NameError:
            import dateparser
            mem = map_dates(mem, time_from)
            
    return mem

def dilate_range(arr, f, p=True):
    '''
    Dilate a range by a certain percentage. If `p` is `True`, `f` represents
    a percentage and thus `arr[0]` will be lowered, and arr[1] raised, 
    by (arr[1] - arr[0])*f. If p=False, arr will be changed by f in absolute
    terms.
    
    `arr`: list-like of length 2 where arr[1] > arr[0]
    `f`: factor by which to dilate the array
    `p`: if `True`, `f` is a percentage of the total range
    
    returns: adjusted np.ndarray of length 2
    '''
    
    arr = np.array(arr)
    
    assert isinstance(p, bool)
    
    if arr.shape[0] != 2 or len(arr.shape) > 1:
        raise ValueError("Param `arr` must be list-like of length 2.")
    if arr[1] < arr[0]:
        raise ValueError("Param `arr` must be sorted, [smaller, larger].")
    if p:
        dilate_by = np.abs(arr[1] - arr[0])*f
    else:
        dilate_by = f
        
    return np.array([arr[0] - dilate_by, arr[1] + dilate_by])

def add_nzmin(arr):
    '''
    Add the non-zero minimum of the array to the array.
    
    arr: np.ndarray with a minimum of zero
    
    returns: adjusted np.ndarray
    '''
    vals = arr.flatten()
    assert vals.min() == 0.0, "Array does not have min of 0."
    nzmin = vals[vals != 0].min()
    return arr + nzmin


def logmp(arr, which=None):
    '''
    Return the logarithm of (`arr` + the minimum non-zero value),
    element-wise. Think of it as np.log1p but instead of 1, 
    add non-zero minimum (hence the "m" in "logmp").

    
    `arr`: np.ndarray with a minimum of zero
    `which`: by default, will take natural logarithm; otherwise,
             which is the base of the logarithm
    
    returns: adjusted np.ndarray
    '''
    
    if isinstance(which, type(None)):
        return np.log(add_nzmin(arr))
    elif which == 10:
        return np.log10(add_nzmin(arr))
    elif which == 2:
        return np.log2(add_nzmin(arr))
    else:
        return np.log(add_nzmin(arr))/np.log(which)


    
def count_lines(path, headchars=None, headonly=False, gz=False, pbar=True):
    '''
    Count the number of lines in a file. If `header_chars` are 
    included, check for line.startswith(`headchars`) until non-header
    line is encountered. 
    
    fname: path to file
    headchars: character to use to check for the header
    headonly: after header lines determined, stop
    pbar: show progress bar if checking non-header lines
    gz: whether or not to use `open` (in 'r' mode) or `gzip.open` in 'rt'
        mode
    
    returns: `dict` of line numbers, each of type `int`
    '''
    
    opener = gzip.open if gz else open
    readmode = 'rt' if gz else 'r'
    head_num = 0
    return_dict = dict()
    
    if not pbar and not headonly:
        stdout = subprocess.run(['wc', '-l', path], stdout=subprocess.PIPE).stdout.decode('utf-8')
        return_dict['total'] = int(stdout.split()[0])
        
    
    with opener(prefix + 'vals/genes.gtf', readmode) as file:
        if not isinstance(headchars, type(None)):
            for line in file:
                if line.startswith(headchars):
                    head_num += 1
                else:
                    break
            return_dict['header'] = head_num
        else:
            head_num = 0
        if not headonly:
            if not pbar:
                nonhead_num = return_dict['total'] - head_num
            else:
                nonhead_num = 0
                for line in tqdm(file):
                    nonhead_num += 1
        return_dict['non-header'] = nonhead_num
        return_dict['total'] = nonhead_num + head_num
        
    return return_dict
    
#### FOR GENERAL FUNCTION INPUT HANDLING
#### GO THROUGH IT AND MAKE SURE IT WORKS
#### THEN REPLACE ALL THE INSTANCES IT STREAMLINES
#### IN ALL FUNCTIONS ACROSS NERO

def replace_none(provided_dict, default_dict):
    '''
    Useful function for replacing values of dict parameter with default
    key-values if the key-value was not provided by the user.
    
    `provided_dict`: dict provided by the user, usually a list of params
                     provided to a certain function within the function
    `default_dict`: the default values to the child function, hardcoded 
                    into the parent function
    
    returns: adjusted `provided_dict` with non-provided default values
    '''
    if isinstance(provided_dict, type(None)):
        provided_dict = default_dict
    else:
        for k in provided_dict:
            default_dict[k] = provided_dict[k]
        provided_dict = default_dict
        
    return provided_dict

def standard_scale(df, by='cols'):
    '''
    Create a standard scale by rows or columns for a given df.
    
    `df`: pd.DataFrame to standardize
    `by`: whether to standardize by columns ('cols') or rows ('rows').
    
    return: new pd.DataFrame with standardized scales
    '''
    df_std = df.copy()
    df_std = df_std.subtract(df_std.min(axis=1), axis=0)
    df_std = df_std.divide(df_std.max(axis=1), axis=0)

    # columns
    df_std = df.copy()
    df_std = df_std.subtract(df_std.min(axis=0), axis=1)
    df_std = df_std.divide(df_std.max(axis=0), axis=1)
 
    return df_std
