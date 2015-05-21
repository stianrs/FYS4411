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

    double runMonteCarloIntegration(int nCycles, int my_rank, int world_size);
    void MonteCarloIntegration(int nCycles, fstream &outfile, int my_rank, int world_size);
    void InvestigateOptimalAlpha();
    void InvestigateOptimalParameters(int my_rank, int world_size);
    void InvestigateVarianceNcycles();
    void InvestigateCPUtime(int my_rank, int world_size);
    void InvestigateTimestep();
    void BlockingFunc(int my_rank, int world_size);
    void OnebodyDensity_ChargeDensity(int my_rank, int world_size);

    void SlaterDeterminant();
    void compute_R_sd(int i);
    void update_D(mat &D_new, const mat &D_old, int i, int selector);
    void SlaterGradient(int i);
    double SlaterLaplacian();
    double SlaterPsi(int particle, int orb_select);
    double Psi_derivative(int particle, int orb_select, int dimension);
    double Psi_laplacian(int particle, int orb_select);


    double factorial_func(int number);
    double Normalization_factor(double GTO_alpha, int i, int j, int k);
    double G_func(double GTO_alpha, int particle, int i, int j, int k);
    double G_derivative(double GTO_alpha, int particle, int orb_select, int dimension, int i, int j, int k);
    double G_laplacian(double GTO_alpha, int particle, int orb_select, int i, int j, int k);

    double GaussianOrbitals(int i, int j);



private:
    double waveFunction(const mat &r);
    void SetParametersAtomType(string AtomType);
    double localEnergy(const mat &r);
    void r_func(const mat &positions);
    void save_positions_func(const mat &r, fstream &outfile);
    double InvestigateOptimalStep();
    void QuantumForce(const mat &r, mat &F);

    void fill_a_matrix();
    double JastrowFactor();

    double SlaterBeryllium();
    double JastrowMultiplicator();
    double psi1s(double &radius);
    double psi2s(double &radius);
    double psi2px(double &x, double &radius);
    double psi2py(double &y, double &radius);
    double psi2pz(double &z, double &radius);


    double ComputeJastrow();
    void fillJastrowMatrix(mat &CorrelationMatrix);
    void compute_R_c(int k);
    void computeJastrowDerivative(int k);
    void computeJastrowLaplacian(int k);
    double computeJastrowEnergy();

    void update_C(mat &CorrelationsMatrix, int k);

    double beta_derivative_Jastrow();
    void findOptimalBeta(int my_rank, int world_size);

    void ReadFile_fillGTO(mat &GTO_mat, string filename);
    void fillGTO();



    string AtomType;

    int nDimensions;

    string energySelector;
    bool optimized_energySolver;
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
    mat r_distanceOld;
    mat a_matrix;
    vec r_radius;
    vec r_radius2;
    vec r_radiusOld;

    vec energy_single;
    vec energySquared_single;
    double variance;
    double averange_r_ij;
    double cpu_time;
    double R_sd;
    double R_c;
    double R;
    double energytermSlaterJastrow;
    double SlaterLaplacianValue;
    double GradientSquared;

    double energy_estimate;
    double E_betaDerivative;

    mat D_down_old;
    mat D_down_new;
    mat D_up_old;
    mat D_up_new;
    mat SlaterGradientOld;
    mat SlaterGradientNew;

    mat C_old;
    mat C_new;
    mat JastrowDerivative;
    mat JastrowDerivativeOld;
    mat JastrowGradient;
    mat JastrowLaplacianNew;
    mat JastrowLaplacianOld;
    double JastrowEnergySum;
    double CrosstermSum;

    mat GTO_helium;
    mat GTO_beryllium;
    mat GTO_neon;
    mat GTO_values;

};


#endif // VMCSOLVER_H
