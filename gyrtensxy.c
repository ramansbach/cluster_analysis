/*
C kernel for computing the gyration tensor entry x,y
*/
double gyrtensxy ( double * posList, int N, int x, int y,double boxlx, double boxly)
{
    double gxy = 0;
    double U,V;
    for (int R = 0; R < N; R++){
        for (int S = 0; S < N; S++){
            V = posList[3*R+x]-posList[3*S+x];
            V = V - boxlx*round(V/boxlx);
            U = posList[3*R+y]-posList[3*S+y];
            U = U - boxly*round(U/boxly);
            gxy = gxy + V*U;
        }
    }
    gxy = gxy/(2*N*N);
    return gxy;

}
