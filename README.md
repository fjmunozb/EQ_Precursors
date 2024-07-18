# EQ_Precursors
Python codes to calculate dv/v and waveform coherence using the stretching method. 

This repository contains the python codes to reproduce the results of the paper
"Days before the strike: precursory waveform decoherence
precedes major strike-slip earthquakes"

Instructions

Download the data from Dryad.org DOI: 10.5061/dryad.n02v6wx5t

Install the noisepy enviroment using environment_noisepy.yml
Download the functions stacking_module.py and dvv_module.py by Jiang and Denolle (2020) Ambient-Noise Seismology Package NoisePy https://github.com/noisepy/NoisePy. 
Run the jupyther notebooks.

* Suppl_run_stretching.ipynb #computes and plots dv/v and WF coherence of a single station pair using one component and filter combination (e.g., STA1-STA2 ZZ 0.1-1.0Hz).
* StrechingRun_COMPUTE_ALL.py #computes dv/v an WF coherence of a single station pair using 9 components at multiple filters (e.g., STA1-STA2 ZZ, ZE,...NZ 0.1-0.5 Hz, 0.1-1.0 Hz, etc.).
* Ridgecrest_All_weighted_statPairs.ipynb, Turkiye_All_weighted_statPairs.ipynb, and Aegean_All_weighted_statPairs.ipynb #plot and compute the by-component weighted dv/v and WF coherence for the three study areas.
* RIDGECREST_plot_BY_FILTER.ipynb, TURKEY_plot_BY_FILTER.ipynb, and AEGEAN_plot_BY_FILTER.ipynb #only plot dv/v and WF coherence by filter and show plots of the by-component weighted dv/v and WF coherence.


Download the entire directory which also contains earthquake data and figures necesary to visualize the plots.

