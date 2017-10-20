#!/bin/bash
#All this script does currently is drops the analyze scripts into the correct folder with the correct templated values
#it can loop over multiple values to set up multiple scripts

RFOLD=/home/mansbac2/coarsegraining/patchy/analysis
SFOLD="\'\/share\/scratch\/ramansbach\/coarsegraining\/hoomd\/patchy\/analyze"

AAAS=(740 185)
SCSCSCS=(0 2)
BBBS=(0 0)

for i in `seq 0 1`; do
AAA=${AAAS[$i]}
SCSCSC=${SCSCSCS[$i]}
BBB=${BBBS[$i]}
CORES=8
sed "s/SSS/$SFOLD\/$AAA-$SCSCSC-$BBB\'/g" analyze_clusters_MPI.py > $RFOLD/$AAA-$SCSCSC-$BBB/analyze_clusters_MPI.py
sed -i "s/AAA/$AAA/g" $RFOLD/$AAA-$SCSCSC-$BBB/analyze_clusters_MPI.py
sed -i "s/SCSCSC/$SCSCSC/g" $RFOLD/$AAA-$SCSCSC-$BBB/analyze_clusters_MPI.py
sed -i "s/BBB/$BBB/g" $RFOLD/$AAA-$SCSCSC-$BBB/analyze_clusters_MPI.py

sed "s/SSS/$SFOLD\/$AAA-$SCSCSC-$BBB\'/g" analyze_cut_mu2.py > $RFOLD/$AAA-$SCSCSC-$BBB/analyze_cut_mu2.py
sed -i "s/AAA/$AAA/g" $RFOLD/$AAA-$SCSCSC-$BBB/analyze_cut_mu2.py
sed -i "s/SCSCSC/$SCSCSC/g" $RFOLD/$AAA-$SCSCSC-$BBB/analyze_cut_mu2.py
sed -i "s/BBB/$BBB/g" $RFOLD/$AAA-$SCSCSC-$BBB/analyze_cut_mu2.py

sed "s/SSS/$SFOLD\/$AAA-$SCSCSC-$BBB\'/g" run-mpi-cutoff-test.sge > $RFOLD/$AAA-$SCSCSC-$BBB/run-mpi-cutoff-test.sge
sed -i "s/AAA/$AAA/g" $RFOLD/$AAA-$SCSCSC-$BBB/run-mpi-cutoff-test.sge
sed -i "s/SCSCSC/$SCSCSC/g" $RFOLD/$AAA-$SCSCSC-$BBB/run-mpi-cutoff-test.sge
sed -i "s/BBB/$BBB/g" $RFOLD/$AAA-$SCSCSC-$BBB/run-mpi-cutoff-test.sge
sed -i "s/CORES/$CORES/g" $RFOLD/$AAA-$SCSCSC-$BBB/run-mpi-cutoff-test.sge

sed "s/SSS/$SFOLD\/$AAA-$SCSCSC-$BBB\'/g" run-mu2-analyze.sge > $RFOLD/$AAA-$SCSCSC-$BBB/run-mu2-analyze.sge
sed -i "s/AAA/$AAA/g" $RFOLD/$AAA-$SCSCSC-$BBB/run-mu2-analyze.sge
sed -i "s/SCSCSC/$SCSCSC/g" $RFOLD/$AAA-$SCSCSC-$BBB/run-mu2-analyze.sge
sed -i "s/BBB/$BBB/g" $RFOLD/$AAA-$SCSCSC-$BBB/run-mu2-analyze.sge
done
