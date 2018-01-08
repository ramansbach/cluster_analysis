/*
C kernel for squashing a larger matrix down to molecule connectivity
*/
void subsquashrng ( double * rng, double * molrng, int dim,
                               int apermol)
{
    int i,j,istart,iend,jstart,jend,k,m;
    int curr;
    for (i = 0; i < dim; i++) {
        for (j = i+1; j < dim; j++) {
            istart = apermol*i;
            iend = apermol*(i+1);
            jstart = apermol*j;
            jend = apermol*(j+1);
            curr = 0;
            for (k = istart; k < iend; k++){
                for (m = jstart; m < jend; m++){
                    if (rng[k*dim*apermol+m] != 0.){
                        curr = 1;
                        //break;
                    }
                }
                //if (curr == 1) {
               //     break;
                //}
            }
            if (curr == 1){
                molrng[dim*i+j] = 1.0;
                molrng[dim*j+i] = 1.0;
                        
                }

    }
    }
}