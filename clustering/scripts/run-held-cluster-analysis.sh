#!/bin/bash
job0=subsamp-AAA-SCSCSC-BBB
jobname=mpi-AAA-02-SCSCSC-150-BBB
#qsub -N $job0 run-gsd-subsample.sge
qsub -hold_jid -N $jobname run-serial-cutoff-test.sge
qsub -hold_jid $jobname run-mu2-analyze.sge
