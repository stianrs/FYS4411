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
void VMCSolver::InvestigateOptimalParameters(){

    SetParametersAtomType(AtomType);

    // fill spin matrix needed if we simulate atoms with more than 2 electrons
    fill_a_matrix();

    nCycles = 100000;
    int nPoints = 100;
    int n = nCycles*nParticles;

    rOld = zeros<mat>(nParticles, nDimensions);
    rNew = zeros<mat>(nParticles, nDimensions);

    double waveFunctionOld = 0;
    double waveFunctionNew = 0;

    double energySum = 0;
    double energySquaredSum = 0;
    double minimumEnergy = 0;

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
    else{
        minimum_Alpha = 3.8;
        maximum_Alpha = 4.1;
        minimum_Beta = 0.05;
        maximum_Beta = 0.15;
    }

    fstream outfile;
    outfile.open("Parameter_Energy_Beryllium.dat", ios::out);

    // run loops over different pairs of alpha and beta
    for(int alphaCounter = 1; alphaCounter < nPoints; alphaCounter++){
        resolution = (maximum_Alpha - minimum_Alpha)/nPoints;
        alpha = minimum_Alpha + resolution*alphaCounter;

        for(int betaCounter = 1; betaCounter < nPoints; betaCounter++){
            resolution = (maximum_Beta - minimum_Beta)/nPoints;
            beta = minimum_Beta + resolution*betaCounter;

            // initial trial positions
            for(int i = 0; i < nParticles; i++) {
                for(int j = 0; j < nDimensions; j++) {
                    rOld(i,j) = stepLength * (ran2(&idum) - 0.5);
                }
            }
            rNew = rOld;

            // loop over Monte Carlo cycles
            for(int cycle = 0; cycle < nCycles; cycle++) {

                // Store the current value of the wave function
                waveFunctionOld = waveFunction(rOld);

                // New position to test
                for(int i = 0; i < nParticles; i++) {
                    for(int j = 0; j < nDimensions; j++) {
                        rNew(i,j) = rOld(i,j) + stepLength*(ran2(&idum) - 0.5);
                    }

                    // Recalculate the value of the wave function
                    waveFunctionNew = waveFunction(rNew);

                    // Check for step acceptance (if yes, update position, if no, reset position)
                    if(ran2(&idum) <= (waveFunctionNew*waveFunctionNew) / (waveFunctionOld*waveFunctionOld)) {
                        for(int j = 0; j < nDimensions; j++) {
                            rOld(i,j) = rNew(i,j);
                            waveFunctionOld = waveFunctionNew;
                        }
                    } else {
                        for(int j = 0; j < nDimensions; j++) {
                            rNew(i,j) = rOld(i,j);
                        }
                    }
                    // update energies
                    deltaE = localEnergy(rNew);
                    energySum += deltaE;
                    energySquaredSum += deltaE*deltaE;
                }
            }

            double energy = energySum/n;
            double energySquared = energySquaredSum/n;
            double variance = (energySquared - (energy*energy))/n;
            outfile << alpha << " " << beta << " " << energy << " " << variance << endl;

            // update parameters if lower energies are found
            if (energy < minimumEnergy){
                minimumEnergy = energy;
                optimalAlpha = alpha;
                optimalBeta = beta;
            }

            // reset energies for next pair of parameters
            energySum = 0.0;
            energy = 0.0;
        }
        cout << "loop num: " << alphaCounter << endl; // counter to see progress in CPU-heavy simulation
    }
    cout << "Optimal Alpha: " << optimalAlpha << " Optimal Beta: " << optimalBeta << " Minimum Energy: " << minimumEnergy << endl;
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
    outfile.open("cpu_time_MPI_1.dat", ios::out);

    int nSimulations = 10;
    int nCycles; // ????  ask for help
    int WorkLoad;

    mat time_values = zeros(nSimulations, 2);

    for(int i=0; i<nSimulations; i++){
        nCycles = 50000 + 1000000*i;

        WorkLoad = nCycles/world_size;

        MonteCarloIntegration(WorkLoad, outfile, my_rank, world_size);
        time_values(i, 0) = nCycles;
        time_values(i, 1) += cpu_time;
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
void VMCSolver::BlockingFunc(){

    nCycles = 1000000;
    int n = (nCycles*nParticles);

    int my_rank, world_size;
    fstream outfile;

    MonteCarloIntegration(nCycles, outfile, my_rank, world_size);

    outfile.open("Blocking_data_xxx.dat", ios::out);
    for(int i=0; i<n; i++){
        outfile << energy_single(i) << " " << energySquared_single(i) << endl;
    }
    outfile.close();
}



// function that run a MC simulation, storing all intermediate positions for all electrons, and write the postions to file
void VMCSolver::OnebodyDensity_ChargeDensity(){

    nCycles = 50000;

    int my_rank, world_size;
    fstream outfile;
    outfile.open("OnebodyDensity_ChargeDensity_xxx.dat", ios::out);
    save_positions = true;
    MonteCarloIntegration(nCycles, outfile, my_rank, world_size);
    outfile.close();
}








