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
    void SetParametersAtomType(string AtomType);
    double localEnergy(const mat &r);
    void r_func(const mat &positions);
    void save_positions_func(const mat &r, fstream &outfile);
    double InvestigateOptimalStep();
    void QuantumForce(const mat &r, mat &F);
    double psi1s(double &radius);
    double psi2s(double &radius);
    double psi2px(double &x, double &radius);
    double psi2py(double &y, double &radius);
    double psi2pz(double &z, double &radius);

    void fill_a_matrix();
    double JastrowFactor();
    double SlaterPsi(const mat &positions, int i, int j);
    void SlaterDeterminant(const mat &positions);
    double compute_R_sd(int k);
    void SlaterGradient(int i);
    double SlaterLaplacian();
    double Psi_first_derivative(int i, int j, int k);
    double Psi_second_derivative(int i, int j);

    double ComputeJastrow();
    void fillJastrowMatrix(mat &CorrelationMatrix);
    void compute_R_c();
    void computeJastrowGradient(int k);
    void computeJastrowLaplacian(int k);
    double computeJastrowEnergy();

    void update_D(mat& D_new, const mat& D_old, int i, int selector);
    void update_C(mat &CorrelationsMatrix, int k);


    string AtomType;

    int nDimensions;

    bool numerical_energySolver;
    bool activate_JastrowFactor;
    bool save_positions;

    //  we fix the time step  for the gaussian deviate
    double timestep;
    // diffusion constant from Schroedinger equation
    double D;
    double stepLength;

    double h;
    double h2;
    time_t idum;

    int charge;
    int nParticles;

    double alpha;
    double beta;

    int nCycles;
    double GreensFunction;

    mat rOld;
    mat rNew;
    mat QForceOld;
    mat QForceNew;
    mat r_distance;
    mat a_matrix;
    vec r_radius;

    vec energy_single;
    vec energySquared_single;
    double variance;
    double averange_r_ij;
    double cpu_time;
    double R_sd;
    double R_c;
    double R;
    double JastrowGradientSquared;
    double JastrowOld;
    double JastrowNew;
    double SlaterLaplacianValue;

    mat D_down_old;
    mat D_down_new;
    mat D_up_old;
    mat D_up_new;
    mat SlaterGradientsOld;
    mat SlaterGradientsNew;

    mat C_old;
    mat C_new;
    mat JastrowGradientNew;
    mat JastrowGradientOld;
    mat JastrowLaplacianNew;
    mat JastrowLaplacianOld;

};


#endif // VMCSOLVER_H
