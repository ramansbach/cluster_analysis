This folder contains scripts demonstrating the use of this clustering package which are mostly employed on an SGE system.

You can use setupscripts.sh to set up a number of different scripts for analysis quite easily--it assumes there are a number of folders run at different parameter steps labeled in the format AAA-SCSCSC-BBB. These are purely for naming convenience, apart from BBB, which also defines the (variable) contact cluster cutoff (see, eg, cluster_analysis_serial.py)

Without using setupscripts.sh, any of the python scripts may be run once their parameters are appropriately set.  The file analyze_clusters_serial.py explains what most of those parameters are.
