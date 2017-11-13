#!/bin/bash
job0=miss-AAA-SCSCSC-BBB
jobname=mpi-AAA-SCSCSC-BBB
#qsub -N $job0 run-misplaced-aroms.sge
qsub -N $jobname run-mpi-cutoff-test.sge
qsub -hold_jid $jobname run-mu2-analyze.sge
