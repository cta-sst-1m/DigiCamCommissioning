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
This script produces ADC distribution out of `zfits` files
 in `DIRECTORY+file_basename+FILE_LIST[:]` taken with HV off and then fits them with a simple gaussian.

Histograms are saved in `output_directory+histo_filename`
and fit results in `output_directory+fit_filename`


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

Example of usage:

Run the full script specifying all options:
`./analyse_hvoff.py -c -p -d /my/data/dir --file_namebase thefilename_ -f 0,1,2 --output_directory
 /my/output/dir --histo_filename ouput_histo_file.npz --fit_filename ouput_fit_file.npz`
 
Run the fit only on saved histograms:
`./analyse_hvoff.py -p /my/output/dir --histo_filename ouput_histo_file.npz --fit_filename ouput_fit_file.npz`
 
Only display the results:
`./analyse_hvoff.py /my/output/dir --histo_filename ouput_histo_file.npz --fit_filename ouput_fit_file.npz`

### `analyse_dark.py` script

```
Usage: analyse_dark.py [options]

Options:
  -h, --help            show this help message and exit
  -q, --quiet           don't print status messages to stdout
  -c, --create_histo    load the ADC with HV ON histograms from file
  -p, --perform_fit     perform fit of ADC from dark run
  -f FILE_LIST, --file_list=FILE_LIST
                        input filenames separated by ','
  --evt_max=EVT_MAX     maximal number of events
  -n N_EVT_PER_BATCH, --n_evt_per_batch=N_EVT_PER_BATCH
                        number of events per batch
  --file_basename=FILE_BASENAME
                        file base name
  -d DIRECTORY, --directory=DIRECTORY
                        input directory
  --histo_filename=HISTO_FILENAME
                        Histogram SPE file name
  --output_directory=OUTPUT_DIRECTORY
                        directory of histo file
  --fit_filename=FIT_FILENAME (NOT IMPLEMENTED)
                        name of fit file with dark
```

This script produces ADC distribution out of `zfits` files
 in `DIRECTORY+file_basename+FILE_LIST[:]` taken with HV on but dark conditions.
 
Histograms are saved in `output_directory+histo_filename`
and fit results in `output_directory+fit_filename`

WARNING: No fit is implemented for the moment. This is where we should develop the fit using convolution
of pdfs for the sliding signal in 4ns.




### `analyse_dark_spe.py` script

This script produces ADC distribution out of `zfits` files
 in `DIRECTORY+file_basename+FILE_LIST[:]` taken with HV on but dark conditions.
 
Histograms are saved in `output_directory+histo_filename`
and fit results in `output_directory+fit_filename`

Fit takes as input the result of the `analyse_hvoff.py` script.




### `analyse_ac_dac_scan.py` script
```
Usage: analyse_ac_dac_scan.py [options]

Options:
  -h, --help            show this help message and exit
  -q, --quiet           don't print status messages to stdout
  -c, --create_histo    load the mpe histo from file
  -t, --create_time_histo
                        load the mpe histo from file
  -k, --create_full_histo
                        load the mpe full histo from file
  -g, --perform_fit_gain
                        perform fit of all mpe to get gain, sigma_e, sigma1
  -p, --perform_fit_mu  perform fit of mpe
  -f FILE_LIST, --file_list=FILE_LIST
                        input filenames separated by ','
  -l SCAN_LEVEL, --scan_level=SCAN_LEVEL
                        list of scans DC level, separated by ',', if only
                        three argument, min,max,step
  -e EVENTS_PER_LEVEL, --events_per_level=EVENTS_PER_LEVEL
                        number of events per level
  --evt_max=EVT_MAX     maximal number of events
  -n N_EVT_PER_BATCH, --n_evt_per_batch=N_EVT_PER_BATCH
                        number of events per batch
  --file_basename=FILE_BASENAME
                        file base name
  -d DIRECTORY, --directory=DIRECTORY
                        input directory
  --histo_filename=HISTO_FILENAME
                        Histogram SPE file name
  --peak_histo_filename=PEAK_HISTO_FILENAME
                        name of peak histo file
  --output_directory=OUTPUT_DIRECTORY
                        directory of histo file
  --fit_filename=FIT_FILENAME
                        name of fit file with MPE
  --input_fit_hvoff_filename=INPUT_HVOFF_FILENAME
                        Input fit file name
  --input_fit_dark_filename=INPUT_DARK_FILENAME
                        Input fit file name
```



## Modules

### `data_treatement` module

