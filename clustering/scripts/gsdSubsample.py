#open a gsd file and write out a subsampled version, keeping only every N timesteps
#useful if you want to be analyzing a shorter trajectory
import gsd.hoomd
import argparse
import time
start = time.time()
parser = argparse.ArgumentParser(description='Subsamble GSD trajectory')
parser.add_argument('fname',metavar='input',type=str,help='trajectory file to be subsampled')
parser.add_argument('ofname',metavar='output',type=str,help='where to write subsampled trajectory file')
parser.add_argument('N',metavar='N',type=int,help='keep frame each N timesteps')

args = parser.parse_args()

traj = gsd.hoomd.open(args.fname)
frame0 = traj[0]
newtraj = gsd.hoomd.open(args.ofname,'wb')
newtraj.append(frame0)
for i in range(args.N,len(traj),args.N):
   s = gsd.hoomd.Snapshot()
   pos = traj[i].particles.position
   s.particles.position = pos
   s.particles.N = len(pos)
   newtraj.append(s)

end = time.time()
print('Subsampling took {0} s.'.format(end-start))
