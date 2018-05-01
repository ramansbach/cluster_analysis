#open a 5-column (from 5 runs) and 1 column set of corrcalc data
#output 5-column with the first (data) column replaced with the 1 column
import argparse,pdb
parser = argparse.ArgumentParser()
parser.add_argument('f5name',metavar='input5',type=str,help='5 column data file')
parser.add_argument('f1name',metavar='input1',type=str,help='1 column data file')
parser.add_argument('outname',metavar='output',type=str,help='where to write output corrected 5 column data file')

args = parser.parse_args()
f5name = args.f5name
f1name = args.f1name
outname = args.outname

f5f = open(f5name)
f1f = open(f1name)
lines5 = f5f.readlines()
lines1 = f1f.readlines()
f5f.close()
f1f.close()
outf = open(outname,'w')
for i in range(len(lines5)):
    
    spline5 = lines5[i].split()
    spline1 = lines1[i].split()
    
    outf.write('{0} {1} {2} {3} {4} {5}\n'.format(spline5[0],spline1[1],spline5[2],spline5[3],spline5[4],spline5[5]))
outf.close()
