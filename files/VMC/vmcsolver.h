#ifndef VMCSOLVER_H
#define VMCSOLVER_H

#include <armadillo>

using namespace arma;

class VMCSolver
{
public:
    VMCSolver();

    void runMonteCarloIntegration();

private:
    double waveFunction(const mat &r, int &wavefunc_selection);
    double waveFunctionAnalytical(const mat &r, int &wavefunc_selection);

    double localEnergy(const mat &r);

    int nDimensions;
    int charge;
    double stepLength;
    int nParticles;

    double h;
    double h2;

    long idum;

    double alpha;
    double beta;

    int nCycles;

    int wavefunc_selection;

    mat rOld;
    mat rNew;
};

#endif // VMCSOLVER_H
