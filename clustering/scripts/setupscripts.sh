#!/bin/bash
#All this script does currently is drops the analyze scripts into the correct folder with the correct templated values
#it can loop over multiple values to set up multiple scripts

for FFF in 10-90 25-75 50-50 75-25 90-10; do
RFOLD=/home/mansbac2/coarsegraining/patchy/analysis/heterogeneous/$FFF
SFOLD="\'\/data\/mansbac2\/coarsegraining\/move_external\/patchy\/heterogeneous\/$FFF"

AAAS=(250)
SCSCSCS=(0 001 02 09)
BBBS=(100 125 150)
for i in `seq 0 1`; do
	AAA=${AAAS[$i]}
for j in `seq 0 3`; do
	SCSCSC=${SCSCSCS[$j]}
for k in `seq 0 2`; do
	BBB=${BBBS[$k]}
	BBBB=$BBB
	BBBA=150
	#BBB=06
	echo $AAA-$SCSCSC-$BBB
	SCRIPTS=(gsdSubsample.py run-gsd-subsample.sge run-held-cluster-analysis.sh analyze_clusters_serial_het.py analyze_cut_mu2_het.py run-mu2-analyze.sge corrdim_timing_het.py run-corrdim-analyze.sge run-serial-cutoff-test.sge)
	#if [ -e "$RFOLD/$AAA-$SCSCSCS-$BBB" ]
	#then
	echo "running $AAA-$SCSCSC-$BBB"
	for script in "${SCRIPTS[@]}"; do
		sed "s/SSS/$SFOLD\/$AAA-02-$SCSCSC-150-$BBB\'/g" $script  > $RFOLD/$AAA-02-$SCSCSC-150-$BBB/$script
		sed -i "s/AAA/$AAA/g" $RFOLD/$AAA-02-$SCSCSC-150-$BBB/$script
		sed -i "s/SCSCSC/\'$SCSCSC\'/g" $RFOLD/$AAA-02-$SCSCSC-150-$BBB/$script
		sed -i "s/BBBA/\'$BBBA\'/g" $RFOLD/$AAA-02-$SCSCSC-150-$BBB/$script
		sed -i "s/BBBB/\'$BBBB\'/g" $RFOLD/$AAA-02-$SCSCSC-150-$BBB/$script
		sed -i "s/BBB/\'$BBB\'/g" $RFOLD/$AAA-02-$SCSCSC-150-$BBB/$script
		sed -i "s/FFF/$FFF/g" $RFOLD/$AAA-02-$SCSCSC-150-$BBB/$script
	done

	#fi
done
done
done
done
