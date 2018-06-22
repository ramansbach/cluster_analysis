#!/bin/bash
job0=subsamp-AAA-SCSCSC-BBB
jobname=mpi-AAA-SCSCSC-BBB
qsub -N $job0 run-gsd-subsample.sge
qsub -hold_jid mpi-750-02-125,mpi-750-2-100,mpi-750-2-125,mpi-750-2-150,mpi-750-6-100,mpi-750-6-150,mpi-750-10-100,mpi-750-10-150,$job0 -N $jobname run-serial-cutoff-test.sge
qsub -hold_jid $jobname run-mu2-analyze.sge
