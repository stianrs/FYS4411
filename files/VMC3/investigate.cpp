/*
This program includes functions that compute different parameters used in the MC solver
and functions to perform spesific collection of data from MC simulations
*/

#include "vmcsolver.h"
#include "lib.h"
#include "investigate.h"
#include <armadillo>
#include <iostream>
#include <time.h>
#include <fstream>
#include <mpi.h>

using namespace arma;
using namespace std;


// algorithm to find a steplength that gives an acceptance ratio about 0.5
double VMCSolver::InvestigateOptimalStep(){

}

// function that run MC simulations with different values of alpha, compute the corresponding
// energy, write to file, and print out the optimal alpha with correponding energy
void VMCSolver::InvestigateOptimalAlpha(){

}

// function that run MC simulations with different values of alpha and beta, compute the corresponding
// energy, write to file, and print out the optimal set of parameters with correponding energy
void VMCSolver::InvestigateOptimalParameters(int my_rank, int world_size){

    SetParametersAtomType(AtomType);

    // fill spin matrix needed if we simulate atoms with more than 2 electrons
    fill_a_matrix();

    nCycles = 100000;
    int nPoints = world_size*12;
    //int n = nCycles*nParticles;

    rOld = zeros<mat>(nParticles, nDimensions);
    rNew = zeros<mat>(nParticles, nDimensions);

    D_down_old = zeros<mat>(nParticles/2, nParticles/2);
    D_down_new = zeros<mat>(nParticles/2, nParticles/2);
    D_up_old = zeros<mat>(nParticles/2, nParticles/2);
    D_up_new = zeros<mat>(nParticles/2, nParticles/2);

    QForceOld = zeros<mat>(nParticles, nDimensions);
    QForceNew = zeros<mat>(nParticles, nDimensions);

    SlaterGradientOld = zeros<mat>(nParticles, nDimensions);
    SlaterGradientNew = zeros<mat>(nParticles, nDimensions);

    C_old = zeros<mat>(nParticles, nParticles);
    C_new = zeros<mat>(nParticles, nParticles);

    JastrowDerivative = zeros<mat>(nParticles, nParticles);
    JastrowDerivativeOld = zeros<mat>(nParticles, nParticles);
    JastrowLaplacianNew = zeros<mat>(nParticles, nParticles);
    JastrowLaplacianOld = zeros<mat>(nParticles, nParticles);


    double energySum = 0;
    double energySquaredSum = 0;

    double minimumEnergy_proc = 0;
    double alpha_proc;
    double beta_proc;

    double acceptCounter = 0;

    double deltaE;
    double resolution;
    double optimalAlpha;
    double optimalBeta;

    double minimum_Alpha;
    double maximum_Alpha;
    double minimum_Beta;
    double maximum_Beta;

    // set range to investigate for different atoms
    if(charge == 2){
        minimum_Alpha = 1.0;
        maximum_Alpha = 3.0;
        minimum_Beta = 0.2;
        maximum_Beta = 0.5;
    }
    else if(charge == 4){
        minimum_Alpha = 3.8;
        maximum_Alpha = 4.1;
        minimum_Beta = 0.05;
        maximum_Beta = 0.15;
    }
    else{
        minimum_Alpha = 8.0;
        maximum_Alpha = 12.0;
        minimum_Beta = 0.05;
        maximum_Beta = 0.15;
    }

    fstream outfile;
    outfile.open("Parameter_Energy_xxx.dat", ios::out);

    int nPoints_proc = nPoints/world_size;

    // run loops over different pairs of alpha and beta
    for(int alphaCounter = nPoints_proc*my_rank; alphaCounter <  nPoints_proc*(my_rank+1); alphaCounter++){
        resolution = (maximum_Alpha - minimum_Alpha)/nPoints;
        alpha = minimum_Alpha + resolution*alphaCounter;
        alpha_proc = alpha;

        for(int betaCounter = 0; betaCounter < nPoints; betaCounter++){
            resolution = (maximum_Beta - minimum_Beta)/nPoints;
            beta = minimum_Beta + resolution*betaCounter;
            beta_proc = beta;

            // initial trial positions
            for(int i = 0; i < nParticles; i++){
                for(int j = 0; j < nDimensions; j++) {
                    rOld(i,j) = GaussianDeviate(&idum)*0.5;
                }
            }
            rNew = rOld;

            // Calculate r_distance and r_radius
            r_func(rNew);

            // Compute everything around Slaterdeterminant
            SlaterDeterminant();
            D_up_old = D_up_new;
            D_down_old = D_down_new;


            //SlaterLaplacianValue = SlaterLaplacian();

            for(int i=0; i<nParticles; i++){
                SlaterGradient(i);
            }
            SlaterGradientOld = SlaterGradientNew;

            // Compute everything about Jastrowfactor
            R_c = 1.0;
            if (activate_JastrowFactor){
                fillJastrowMatrix(C_new);
                C_old = C_new;

                JastrowDerivativeOld = JastrowDerivative; // Probably not necessary JastrowGradientOld
                JastrowLaplacianOld = JastrowLaplacianNew;

                for(int i=0; i<nParticles; i++){
                    computeJastrowDerivative(i);
                    computeJastrowLaplacian(i);
                }
                JastrowDerivativeOld = JastrowDerivative;
                JastrowLaplacianOld = JastrowLaplacianNew;
            }

            // Compute quantum force initial state
            QuantumForce(rNew, QForceNew);
            QForceOld = QForceNew;

            acceptCounter = 0;

            // loop over Monte Carlo cycles
            for(int cycle = 0; cycle < nCycles; cycle++) {

                // New position to test
                for(int i = 0; i < nParticles; i++) {
                    for(int j = 0; j < nDimensions; j++) {
                        rNew(i,j) = rOld(i,j) + GaussianDeviate(&idum)*sqrt(timestep)+QForceOld(i,j)*timestep*D;
                    }

                    // Update r_distance and r_radius
                    r_func(rNew);

                    compute_R_sd(i);

                    if (activate_JastrowFactor){
                        update_C(C_new, i);
                        computeJastrowDerivative(i);
                        computeJastrowLaplacian(i);
                        compute_R_c(i);
                    }

                    QuantumForce(rNew, QForceNew);

                    //  we compute the log of the ratio of the greens functions to be used in the
                    //  Metropolis-Hastings algorithm
                    GreensFunction = 0.0;
                    for (int j=0; j < nDimensions; j++) {
                        GreensFunction += 0.5*(QForceOld(i,j)+QForceNew(i,j))*
                                (D*timestep*0.5*(QForceOld(i,j)-QForceNew(i,j))-rNew(i,j)+rOld(i,j));
                    }
                    GreensFunction = exp(GreensFunction);

                    R = R_sd*R_c;

                    // The Metropolis test is performed by moving one particle at the time
                    if(ran2(&idum) <= GreensFunction*R*R){
                        for(int j = 0; j < nDimensions; j++){
                            rOld(i,j) = rNew(i,j);
                        }

                        QForceOld = QForceNew;
                        C_old = C_new;

                        SlaterGradientOld = SlaterGradientNew; // SlaterGradientsOld probably totally unesscesary
                        JastrowDerivativeOld = JastrowDerivative;
                        JastrowLaplacianOld = JastrowLaplacianNew;


                        // Recalculate Slater matrices D
                        if(i<nParticles/2){
                            update_D(D_up_new, D_up_old, i, 0);
                            D_up_old = D_up_new;
                        }
                        else{
                            update_D(D_down_new, D_down_old, i, 1);
                            D_down_old = D_down_new;
                        }
                        acceptCounter += 1;

                        // compute energies
                        deltaE = localEnergy(rNew);
                        energySum += deltaE;
                        energySquaredSum += deltaE*deltaE;
                    }

                    else {
                        for(int j=0; j<nDimensions; j++) {
                            rNew(i,j) = rOld(i,j);
                        }

                        r_func(rOld);

                        QForceNew = QForceOld;
                        C_new = C_old;

                        SlaterGradientNew = SlaterGradientOld; // SlaterGradientsOld probably totally unesscesary
                        JastrowDerivative = JastrowDerivativeOld;
                        JastrowLaplacianNew = JastrowLaplacianOld;

                        D_up_new = D_up_old;
                        D_down_new = D_down_old;

                    }
                }
            }
            double energy_mean_proc = energySum/acceptCounter;
            double energySquared_mean = energySquaredSum/acceptCounter;

            //cout << alpha << " " << beta << " " << energy_mean_proc <<  endl;

            // update parameters if lower energies are found
            if (energy_mean_proc < minimumEnergy_proc){
                minimumEnergy_proc = energy_mean_proc;
                optimalAlpha = alpha_proc;
                optimalBeta = beta_proc;
            }

            // reset energies for next pair of parameters
            energySum = 0.0;
            energy_mean_proc = 0.0;
            energySquared_mean = 0.0;
        }
        cout << "loop num: " << alphaCounter << endl; // counter to see progress in CPU-heavy simulation
    }
    MPI_Barrier(MPI_COMM_WORLD);
    cout << "Optimal Alpha: " << optimalAlpha << " Optimal Beta: " << optimalBeta << " Minimum Energy: " << minimumEnergy_proc << endl;
    outfile.close();
}


// function that run MC simulations with different number of nCycles, compute the energy for every simulation, and write to file
void VMCSolver::InvestigateVarianceNcycles(){
    int nSimulations = 40;
    double variance;

    int my_rank, world_size;
    fstream outfile;
    outfile.open("Variance_nSampels_xxx.dat", ios::out);

    for(int i=0; i < nSimulations; i++){
        nCycles = 5000 + 1000*i;
        int n = (nCycles*nParticles);

        MonteCarloIntegration(nCycles, outfile, my_rank, world_size);
        double energy = sum(energy_single)/n;
        outfile << nCycles << " " << variance << " "  << energy << endl;
    }
    outfile.close();
}


// function that run MC simulations for all combinations of wavefunc and energySolver for with different number of nCycles,
// and store all relevant data for each simulation, and write the data to file
void VMCSolver::InvestigateCPUtime(int my_rank, int world_size){

    fstream outfile;
    outfile.open("cpu_time_MPI_2_new.dat", ios::out);

    int nSimulations = 20;
    int nCycles; // ????  ask for help
    double t_start = 0.0;
    double t_stop = 0.0;

    mat time_values = zeros(nSimulations, 2);

    for(int i=0; i<nSimulations; i++){
        t_start = 0.0;
        t_stop = 0.0;

        nCycles = 50000 + 1000000*i;

        t_start = MPI_Wtime();
        MonteCarloIntegration(nCycles, outfile, my_rank, world_size);
        t_stop = MPI_Wtime();
        time_values(i, 0) = nCycles;
        time_values(i, 1) += (t_stop - t_start);

        MPI_Barrier(MPI_COMM_WORLD);
    }
    outfile << time_values << endl;
    outfile.close();
}


// function that run MC simulations for different timesteps, and store all relevant data for each simulation, and write the data to file
void VMCSolver::InvestigateTimestep(){

    nCycles = 10000000;

    int nSimulations = 10;

    int my_rank, world_size;
    fstream outfile;
    outfile.open("timestep_dependence_xxx.dat", ios::out);

    int counter = 0;
    for(int i=0; i < nSimulations; i++){
        timestep = 0.001 + 0.001*i;

        int n = (nCycles*nParticles);

        MonteCarloIntegration(nCycles, outfile, my_rank, world_size);

        double energy = sum(energy_single)/n;
        outfile << timestep << " " << energy << " " << averange_r_ij << " " << cpu_time <<  " " << nCycles << endl;
        cout << "counter: " << counter << endl; // counter to see progress in CPU-heavy simulation
        counter++;
    }
    outfile.close();
}


// function that run a MC simulation, compute and store all intermediate energy, and write the energy to file
void VMCSolver::BlockingFunc(int my_rank, int world_size){

    int nCycles = 10000000;

    fstream outfile;
    MonteCarloIntegration(nCycles, outfile, my_rank, world_size);

    outfile.open("Blocking_data_Be2_new.dat", ios::out);
    for(int i=0; i < nCycles*nParticles; i++){
        outfile << energy_single(i) << " " << energySquared_single(i) << endl;
    }
    outfile.close();
}


// function that run a MC simulation, storing all intermediate positions for all electrons, and write the postions to file
void VMCSolver::OnebodyDensity_ChargeDensity(int my_rank, int world_size){

    nCycles = 5000000;

    fstream outfile;
    outfile.open("OnebodyDensity_ChargeDensity_beryllium_corrected_jastrow_new.dat", ios::out);
    save_positions = true;
    MonteCarloIntegration(nCycles, outfile, my_rank, world_size);
    outfile.close();
}


// Function to run over different inter-molecular distances and compute corresponding energy with optimal beta, write results to file
void VMCSolver::R_dependence_molecules(int my_rank, int world_size){
    nCycles = 1000000;

    int R_iter = 40;
    fstream outfile;
    outfile.open("R_dependence_molecule_H2_minus.dat", ios::out);

    for(int i=0; i<R_iter; i++){
        R_molecule = 0.2 + 0.2*i;
        findOptimalBeta(my_rank, world_size);
        MonteCarloIntegration(nCycles, outfile, my_rank, world_size);
        cout << R_molecule << " " << beta  << " " << energy_estimate << endl;
        outfile << R_molecule << " " << beta  << " " << energy_estimate << endl;
    }
    outfile.close();
}







