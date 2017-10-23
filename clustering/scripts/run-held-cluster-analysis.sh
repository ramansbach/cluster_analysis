#!/bin/bash
jobname=mpi-AAA-SCSCSC-BBB
qsub -N $jobname run-mpi-cutoff-test.sge
qsub -hold_jid $jobname run-mu2-analyze.sge
