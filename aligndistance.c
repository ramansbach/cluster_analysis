double aligndistance (double * dists, double * distsA, double * distsB, 
                      double * x, double * y, int ats) { 
    int i,j, k, dind = 0;
    for (i = 0; i < ats; i++) {
        for (j = 0; j < ats; j++) {
            dists[dind] = (x[3 * i] - y[3 * j]) * (x[3 * i] - y[3 * j])
            + (x[3 * i + 1] - y[3 * j + 1]) * (x[3 * i + 1] - y[3 * j + 1])
            + (x[3 * i + 2] - y[3 * j + 2]) * (x[3 * i + 2] - y[3 * j + 2]);
            distsA[dind] = i;
            distsB[dind] = j;
            dind++;
        }    
    }
    double mind = 10000.0;
    int mindi, mindj;
    
    for (k = 0; k < ats * ats; k++) {
        if (dists[k] < mind){
            mind = dists[k];
            mindi = distsA[k];
            mindj = distsB[k];
            
        }
    }
    double mind2 = 10000.0;
    int mind2i, mind2j;
    for (k = 0; k < ats * ats; k++){
        if ((dists[k] < mind2) && (distsA[k] != mindi) && (distsB[k] != mindj))
        {
            mind2 = dists[k];
            mind2i = distsA[k];
            mind2j = distsB[k];
        }
    }
    double mind3 = 10000.0;
    for (k = 0; k < ats * ats; k++){
        if ((dists[k] < mind3) && (distsA[k] != mindi) && (distsB[k] != mindj) 
        && (distsA[k] != mind2i) && (distsB[k] != mind2j)){
            mind3 = dists[k];
        }
    }
    return mind3;
}