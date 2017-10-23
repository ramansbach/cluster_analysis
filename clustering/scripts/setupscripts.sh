#!/bin/bash
#All this script does currently is drops the analyze scripts into the correct folder with the correct templated values
#it can loop over multiple values to set up multiple scripts

RFOLD=/home/mansbac2/coarsegraining/patchy/analysis
SFOLD="\'\/share\/scratch\/ramansbach\/coarsegraining\/hoomd\/patchy\/analyze"

AAAS=(0 0 0 0 37 37 37 37 37 185 185 185 185 185 370 370 370 370 370 740 740 740 740 740)
SCSCSCS=(02 2 4 10 0 02 2 4 10 0 02 2 4 10 0 02 2 4 10 0 02 2 4 10)
BBBS=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)

for i in `seq 0 23`; do
	AAA=${AAAS[$i]}
	SCSCSC=${SCSCSCS[$i]}
	BBB=${BBBS[$i]}
	CORES=8
	echo $i-$AAA-$SCSCSC-$BBB
	SCRIPTS=(run-held-cluster-analysis.sh analyze_clusters_MPI.py analyze_cut_mu2.py run-mu2-analyze.sge fix_misplaced_aroms.py run-misplaced-aroms.sge)
	for script in "${SCRIPTS[@]}"; do
		sed "s/SSS/$SFOLD\/$AAA-$SCSCSC-$BBB\'/g" $script  > $RFOLD/$AAA-$SCSCSC-$BBB/$script
		sed -i "s/AAA/$AAA/g" $RFOLD/$AAA-$SCSCSC-$BBB/$script
		sed -i "s/SCSCSC/\'$SCSCSC\'/g" $RFOLD/$AAA-$SCSCSC-$BBB/$script
		sed -i "s/BBB/$BBB/g" $RFOLD/$AAA-$SCSCSC-$BBB/$script
	done

	sed "s/SSS/$SFOLD\/$AAA-$SCSCSC-$BBB\'/g" run-mpi-cutoff-test.sge > $RFOLD/$AAA-$SCSCSC-$BBB/run-mpi-cutoff-test.sge
	sed -i "s/AAA/$AAA/g" $RFOLD/$AAA-$SCSCSC-$BBB/run-mpi-cutoff-test.sge
	sed -i "s/SCSCSC/\'$SCSCSC\'/g" $RFOLD/$AAA-$SCSCSC-$BBB/run-mpi-cutoff-test.sge
	sed -i "s/BBB/$BBB/g" $RFOLD/$AAA-$SCSCSC-$BBB/run-mpi-cutoff-test.sge
	sed -i "s/CORES/$CORES/g" $RFOLD/$AAA-$SCSCSC-$BBB/run-mpi-cutoff-test.sge

done
