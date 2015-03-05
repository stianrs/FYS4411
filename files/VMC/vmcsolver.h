#ifndef VMCSOLVER_H
#define VMCSOLVER_H

#include <armadillo>
#include <fstream>
#include <iostream>

using namespace std;
using namespace arma;

class VMCSolver
{
public:
    VMCSolver();

    void runMonteCarloIntegration(int nCycles);
    void MonteCarloIntegration(int nCycles, fstream &outfile);
    void InvestigateOptimalAlpha();
    void InvestigateOptimalParameters();
    void InvestigateVarianceNcycles();
    void InvestigateCPUtime();
    void InvestigateTimestep();
    void BlockingFunc();
    void OnebodyDensity_ChargeDensity();


private:
    double waveFunction(const mat &r);
    double localEnergy(const mat &r);
    void r_func(const mat &positions);
    void save_positions_func(const mat &r, fstream &outfile);
    double InvestigateOptimalStep();
    void QuantumForce(const mat &r, mat &F);
    double psi1s(double &r);
    double psi2s(double &r);
    void fill_a_matrix();
    double JastrowFactor();
    double SlaterDeterminant();

    int nDimensions;
    int charge;
    int nParticles;
    double stepLength;

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
    bool deactivate_ImportanceSampling;
    bool save_positions;

    mat rOld;
    mat rNew;
    mat QForceOld;
    mat QForceNew;
    mat r_distance;
    mat a_matrix;
    vec r_centre;

    vec energy_single;
    vec energySquared_single;
    double variance;
    double averange_r_ij;
    double time;

};


#endif // VMCSOLVER_H
