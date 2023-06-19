# clue: Efficient Pooled Single-cell Sequencing

Clue is a framework for efficient single-cell sequencing using genetic multiplexing. Cells from genetically distinct individuals are mixed in multiple pools such that their presence or absence in a given pool is fully deterministic of which individual they come from. Pooling in this fashion and using a genotype-free demultiplexing algorithm obviates the need for orthogonal genotyping to demultiplex data.

**19JUN23**: Repo is not complete yet. Notebooks and files are being validated on a fresh machine and will be progressively uploaded over the coming weeks/months as they are validated that they run to completion with no errors. Then data will be uploaded to GEO so that all analyses are reproducible.

## File Organization

The file organization for this project:

 1) The **data files** contain raw or processed data (separated values files, txt, h5, h5ad, etc.) housed within `/path/to/mountpoint/`, wherever you decide to mount the data in your compute environment. Data files are provided on GEO.

 2) The **scripts and notebooks** import and export the data files for processing, analysis, and figure generation. At the top of mostly every notebook you will be able to set the `/paht/to/mountpoint/`. Scripts and notebooks are provided in this repository. 

To find out whether data files are raw or generated during analysis processing, check `describe_files.csv` within this respository.

## Environment Setup

All places where you see `clue_test` may just be replaced with `clue` for setting up your files if you'd like to reproduce analyses. The word `test` was just included when I was testing that all the notebooks run in a freshly-prepared conda environment and making sure I had the minimum necessary data files and notebooks to reproduce all the analyses in the paper.

All notebooks were tested and ran to completion on an Amazon Web Services (AWS) Elastic Compute Cloud (EC2) instance, with a Ubuntu 18.04.1 x86_64 Amazon Machine Image (AMI) software configuration using a m5a.4xlarge (16 vCPU, 64 GiB Memory) instance type. Miniconda (py38_4.9.2) was installed on the instance using a bash script, and then mamba and Jupyter were installed within the base environment:

```
(base) ubuntu@ip:/$ conda install -c conda-forge jupyterlab; conda install mamba;
...
...
...
(base) ubuntu@ip:/$ jupyter --version; mamba --version;
jupyter core     : 4.7.0
jupyter-notebook : 6.1.6
qtconsole        : not installed
ipython          : 7.19.0
ipykernel        : 5.4.2
jupyter client   : 6.1.11
jupyter lab      : 3.0.2
nbconvert        : 6.0.7
ipywidgets       : 7.6.3
nbformat         : 5.0.8
traitlets        : 5.0.5
mamba 0.15.3
conda 4.9.2
```

Notebooks were run within a conda environment `clue_test` created from the clue_test.yml using the below procedure:

```
(base) ubuntu@ip:/$ conda create -n clue_test
(base) ubuntu@ip:/$ conda activate clue_test
(clue_test) ubuntu@ip:/$ mamba env update --file clue_test.yml --prune
```

The conda environment was set as a Jupyter Lab kernel. Upon running ipykernel (JUN2023), I encountered an error:

```
ImportError: cannot import name 'generator_to_async_generator'
```

And followed the instructions at [ImportError: cannot import name 'generator_to_async_generator' #11270](https://github.com/ipython/ipython/issues/11270#issuecomment-427448691):
```
(clue_test) ubuntu@ip:/$ pip uninstall -y ipython prompt_toolkit
(clue_test) ubuntu@ip:/$ pip install ipython prompt_toolkit
(clue_test) ubuntu@ip:/$ python -m ipykernel install --user --name clue_test --display-name "clue_test"
```

Install the `clue_helper` package within this repo:

```
(clue_test) ubuntu@ip:/$ cd /path/to/clue/clue_helper/
(clue_test) ubuntu@ip:/$ pip install .
```

**TODO 19JUN23: Figure out where pandas was updated. Document it here, then re-install pandas==1.0.5.**
```
(clue_test) ubuntu@ip:/$ mamba install pandas==1.0.5
```

## Reproducing Analyses with Notebooks

### Metadata File

Throughout the notebooks, I make use of a Python pickle called `meta.pkl`, which is available on GEO. It was created using `create_metadata.ipynb`. The pickle file was updated multiple times throughout the analysis and makes use of processed files created from other notebooks. Therefore, it's a bit circular in that you need `meta.pkl` to run those notebooks cleanly but their outputs are used to create meta.pkl. To reproduce results from individual notebooks, I recommend using the `meta.pkl` provided on GEO outright, but I provided `create_metadata.ipynb` to show how it was created.

### Other Files

Many other files are created throughout the notebooks. I provide them in GEO and I comment out the lines within the notebook where they were exported to avoid overwriting. To find out whether data files are raw or generated during analysis processing, check `describe_files.csv` within this respository.

