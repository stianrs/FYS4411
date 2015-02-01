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
    double localEnergy(const mat &r, int &energySolver_selection, int &wavefunc_selection);
    double r12_func(const mat &r);
    double InvestigateOptimalAlpha();

    int nDimensions;
    int charge;
    int Z;
    double stepLength;
    int nParticles;

    double h;
    double h2;

    long idum;

    double alpha;
    double beta;

    int nCycles;

    int wavefunc_selection;
    int energySolver_selection;

    mat rOld;
    mat rNew;
};

#endif // VMCSOLVER_H
