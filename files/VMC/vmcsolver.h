#ifndef VMCSOLVER_H
#define VMCSOLVER_H

#include <armadillo>


using namespace arma;

class VMCSolver
{
public:
    VMCSolver();

    void runMonteCarloIntegration(int nCycles);
    void MonteCarloIntegration(int nCycles, vec &energy_single, vec &energySquared_single, double &variance, double &averange_r12, double &time);
    void importanceMonteCarloIntegration(int nCycles, vec &energy_single, vec &energySquared_single, double &variance, double &averange_r12, double &time);
    void InvestigateOptimalAlpha();
    void InvestigateOptimalParameters();
    void InvestigateVarianceNcycles();
    void InvestigateCPUtime();
    void InvestigateTimestep();
    void BlockingFunc();
    void OnebodyDensity_ChargeDensity();

private:
    double waveFunction(const mat &r, int &wavefunc_selection);
    double localEnergy(const mat &r, int &energySolver_selection, int &wavefunc_selection);
    void r12_func(const mat &r);
    double InvestigateOptimalStep();
    double QuantumForce(const mat &r, mat F);
    double psi1s(r);
    double psi2s(r);
    void fill_a_matrix();

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

    //  we fix the time step  for the gaussian deviate
    double timestep;
    // diffusion constant from Schroedinger equation
    double D;
    double GreensFunction;


    int wavefunc_selection;
    int energySolver_selection;
    int activate_ImportanceSampling;

    mat rOld;
    mat rNew;
    mat QForceOld;
    mat QForceNew;
    mat r;
    mat a_matrix;
};


#endif // VMCSOLVER_H
