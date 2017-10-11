/*
C kernel for computing the distance metric between two numpy arrays
of length ats
*/
double conoptdistance (double * x, double * y, int ats)
{
    int i,j;
    double d;
    double mind = 10000.0;
    for (i = 0; i < ats; i++) {
       for (j = 0; j < ats; j++) {
           d = (x[3*i] - y[3*j]) * (x[3*i] - y[3*j]) 
               + (x[3*i + 1] - y[3*j + 1]) * (x[3*i + 1] - y[3*j + 1]) 
               + (x[3*i + 2] - y[3*j + 2]) * (x[3*i + 2] - y[3*j + 2]);
           if (d < mind) {
               mind = d;            
           }
       }     
    }
    
    return mind;
}