#!/bin/bash
#All this script does currently is drops the analyze scripts into the correct folder with the correct templated values
#it can loop over multiple values to set up multiple scripts

RFOLD=/home/mansbac2/coarsegraining/patchy/analysis
SFOLD="\'\/share\/scratch\/ramansbach\/coarsegraining\/hoomd\/patchy\/simulations\/offset_little"

AAAS=(0 37 185 370 740)
SCSCSCS=(02 2 4 10 0)
BBBS=(06)
HOSTS=(0 1 2)
HOSTI=0
for i in `seq 0 4`; do
	AAA=${AAAS[$i]}
for j in `seq 0 4`; do
	SCSCSC=${SCSCSCS[$j]}
#for k in `seq 0 1`; do
	#BBB=${BBBS[$k]}
	BBB=06
	CORES=4
	echo $AAA-$SCSCSC-$BBB
	SCRIPTS=(fractald_extract.py run-fd-extract.sge run-held-cluster-analysis.sh analyze_clusters_MPI.py analyze_cut_mu2.py run-mu2-analyze.sge fix_misplaced_aroms.py run-misplaced-aroms.sge corrdim_timing.py run-corrdim-analyze.sge)
	#if [ -e "$RFOLD/$AAA-$SCSCSCS-$BBB" ]
	#then
	echo "running $AAA-$SCSCSC-$BBB"
	for script in "${SCRIPTS[@]}"; do
		sed "s/SSS/$SFOLD\/$AAA-$SCSCSC-$BBB\'/g" $script  > $RFOLD/$AAA-$SCSCSC-$BBB/$script
		sed -i "s/AAA/$AAA/g" $RFOLD/$AAA-$SCSCSC-$BBB/$script
		sed -i "s/SCSCSC/\'$SCSCSC\'/g" $RFOLD/$AAA-$SCSCSC-$BBB/$script
		sed -i "s/BBB/\'$BBB\'/g" $RFOLD/$AAA-$SCSCSC-$BBB/$script
	done

	sed "s/SSS/$SFOLD\/$AAA-$SCSCSC-$BBB\'/g" run-mpi-cutoff-test.sge > $RFOLD/$AAA-$SCSCSC-$BBB/run-mpi-cutoff-test.sge
	sed -i "s/AAA/$AAA/g" $RFOLD/$AAA-$SCSCSC-$BBB/run-mpi-cutoff-test.sge
	sed -i "s/SCSCSC/\'$SCSCSC\'/g" $RFOLD/$AAA-$SCSCSC-$BBB/run-mpi-cutoff-test.sge
	sed -i "s/BBB/\'$BBB\'/g" $RFOLD/$AAA-$SCSCSC-$BBB/run-mpi-cutoff-test.sge
	sed -i "s/CORES/$CORES/g" $RFOLD/$AAA-$SCSCSC-$BBB/run-mpi-cutoff-test.sge
	sed -i "s/HOST/${HOSTS[$HOSTI]}/g" $RFOLD/$AAA-$SCSCSC-$BBB/run-mpi-cutoff-test.sge
	echo "host is ${HOSTS[$HOSTI]}"
	let HOSTI++
	let HOSTI=HOSTI%3
	#fi
#done
done
done
