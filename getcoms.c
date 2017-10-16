void getcoms( double * pos, double * coms, double * masslist, int beads, int mols) 
    {
    int i,j;
    double x,y,z,X,Y,Z,M;
    
    for (i = 0; i < mols; i++) // loop through each molecule
        {
            X = 0;
            Y = 0;
            Z = 0;
            M = 0;
            for (j = 0; j < beads; j++)
                {
                    x = masslist[j] * pos[mols*i + 3*j];
                    y = masslist[j] * pos[mols*i + 3*j+1];
                    z = masslist[j] * pos[mols*i + 3*j+2];
                    X+=x;
                    Y+=y;
                    Z+=z;
                    M += masslist[j];
                }
            X/=M;
            Y/=M;
            Z/=M;
            coms[3*i] = X;
            coms[3*i + 1] = Y;
            coms[3*i + 2] = Z;
        }    
    return;
    }