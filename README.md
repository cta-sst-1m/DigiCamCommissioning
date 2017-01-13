# DigiCamCommissioning
A repository allowing to perform the various SST1M 
DigiCam commissionning tasks.
It is build on the `ctapipe` pipeline

## Structure of the repository

In the base folder one can find the high level 
scripts which allow for example to evaluate the
 gain from low light data or which allow to 
 evaluate LED calibration, etc... as well as a set
  of modules holding the low level functionnalities 

The module `data_treatement` contains algorithms 
to various type of histograms out of the data.

The module `spectra_fit` contains high level fits 
definitions.

The module `utils` contains low level functionnalities
such as histogram class, peak finder, plotting helpers,
etc...

## Calibration scripts

### `analyse_hvoff.py` script
This script .....


```
Usage: ./analyse_hvoff.py [options]


Options:
  -h, --help            show this help message and exit
  -q, --quiet           don't print status messages to stdout
  -c, --create_histo    create the histogram
  -p, --perform_fit     perform fit of ADC with HV OFF
  -f FILE_LIST, --file_list=FILE_LIST
                        input filenames separated by ','
  --evt_max=EVT_MAX     maximal number of events
  -n N_EVT_PER_BATCH, --n_evt_per_batch=N_EVT_PER_BATCH
                        number of events per batch
  --cts_sector=CTS_SECTOR
                        Sector covered by CTS
  --file_basename=FILE_BASENAME
                        file base name
  -d DIRECTORY, --directory=DIRECTORY
                        input directory
  --histo_filename=HISTO_FILENAME
                        Histogram ADC HV OFF file name
  --output_directory=OUTPUT_DIRECTORY
                        directory of histo file
  --fit_filename=FIT_FILENAME
                        name of fit file with ADC HV OFF
```
### `analyse_hvon.py` script

### `bla` script

## Modules

### `data_treatement` module

