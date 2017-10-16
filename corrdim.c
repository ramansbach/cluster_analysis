void corrdim ( double * epss, double * ce, double * distsq, int Nepss, int Nd2)
    {
    int i,k;
    for (i = 0; i < Nd2; i++)
        {
        for (k = 0; k < Nepss; k++)
            {
            if (distsq[i] <= epss[k]*epss[k])
                {
                ce[k]++;
                }
            }    
        }
    }