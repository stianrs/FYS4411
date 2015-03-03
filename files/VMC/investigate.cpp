
#include "vmcsolver.h"
#include "lib.h"
#include "investigate.h"

#include <armadillo>
#include <iostream>
#include <time.h>
#include <fstream>

using namespace arma;
using namespace std;

double VMCSolver::InvestigateOptimalStep()
{
    fill_a_matrix();

    rOld = zeros<mat>(nParticles, nDimensions);
    rNew = zeros<mat>(nParticles, nDimensions);

    double waveFunctionOld = 0;
    double waveFunctionNew = 0;

    double energySum = 0;
    double deltaE;

    double ratioTrial = 0.0;
    double ratio = 1.0;
    int counter = 0;
    double acceptCounter = 0;

    int stepCycles = floor(nCycles/100);

    double optimalStep = 1.0;
    double stepTrialStart = 0.1;
    double maxStepLength = 10.0;
    int StepTrials = floor(maxStepLength/stepTrialStart);

    // initial trial positions
    for(int i = 0; i < nParticles; i++) {
        for(int j = 0; j < nDimensions; j++) {
            rOld(i,j) = stepLength * (ran2(&idum) - 0.5);
        }
    }
    rNew = rOld;

    for(int stepCount = 0; stepCount < StepTrials; stepCount++){
        stepLength = stepTrialStart + stepCount*stepTrialStart;

        // loop over Monte Carlo cycles
        for(int cycle = 0; cycle < stepCycles; cycle++){

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
                    acceptCounter += 1;
                    counter += 1;
                }
                else {
                    for(int j = 0; j < nDimensions; j++) {
                        rNew(i,j) = rOld(i,j);
                    }
                    counter += 1;
                }

                // update energies
                deltaE = localEnergy(rNew);
                energySum += deltaE;
            }
        }

        ratioTrial = acceptCounter/counter;
        if (abs(ratioTrial-0.5) < abs(ratio-0.5)){
            ratio = ratioTrial;
            optimalStep = stepLength;
        }
        else{
            break;
        }
        counter = 0;
        acceptCounter = 0;
    }
    cout << "Optimal step length: " << optimalStep << "  Acceptance ratio: " << ratio << endl;
    return optimalStep;
}








void VMCSolver::InvestigateOptimalAlpha(){
    rOld = zeros<mat>(nParticles, nDimensions);
    rNew = zeros<mat>(nParticles, nDimensions);

    double waveFunctionOld = 0;
    double waveFunctionNew = 0;

    double energySum = 0;
    double energySquaredSum = 0;

    double deltaE;

    int nPoints = 21;
    double resolution;

    fstream outfile;
    outfile.open("Alpha_Energy.dat", ios::out);

    for(int alphaCounter = 0; alphaCounter < nPoints; alphaCounter++){
        resolution = 3.0/nPoints;
        alpha = resolution*alphaCounter;

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

        double energy = energySum/(nCycles * nParticles);
        cout << "Alpha: " << alpha << "  Energy: " << energy << endl;
        outfile << alpha << " " << energy << endl;

        energySum = 0.0;
        energy = 0.0;
    }
    outfile.close();
}





void VMCSolver::InvestigateOptimalParameters(){

    fill_a_matrix();

    rOld = zeros<mat>(nParticles, nDimensions);
    rNew = zeros<mat>(nParticles, nDimensions);

    double waveFunctionOld = 0;
    double waveFunctionNew = 0;

    double energySum = 0;
    double energySquaredSum = 0;

    double deltaE;

    int nPoints = 20;
    double resolution;

    double optimalAlpha;
    double optimalBeta;
    double minimumEnergy = 0;

    //double minimum_Alpha = 1.0;
    //double maximum_Alpha = 3.0;
    //double minimum_Beta = 0.2;
    //double maximum_Beta = 0.5;

    double minimum_Alpha = 3.8;
    double maximum_Alpha = 4.1;
    double minimum_Beta = 0.05;
    double maximum_Beta = 0.15;

    nCycles = 100000;

    fstream outfile;
    outfile.open("Parameter_Energy_Beryllium.dat", ios::out);

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

            double energy = energySum/(nCycles * nParticles);
            double energySquared = energySquaredSum/(nCycles * nParticles);
            double variance = (energySquared - (energy*energy))/(nCycles * nParticles);
            //cout << "Alpha: " << alpha << " Beta: " << beta << " Energy: " << energy << endl;
            outfile << alpha << " " << beta << " " << energy << " " << variance << endl;

            if (energy < minimumEnergy){
                minimumEnergy = energy;
                optimalAlpha = alpha;
                optimalBeta = beta;
            }

            energySum = 0.0;
            energy = 0.0;
        }
        cout << "loop num: " << alphaCounter << endl;
    }
    cout << "Optimal Alpha: " << optimalAlpha << " Optimal Beta: " << optimalBeta << " Minimum Energy" << minimumEnergy << endl;
    outfile.close();
}


void VMCSolver::InvestigateVarianceNcycles(){
    int nSimulations = 40;
    double variance;
    int nCycles;

    int n = (nCycles*nParticles);

    double averange_r_ij;
    double time;


    fstream outfile;
    outfile.open("Variance_nSampels_no_imp.dat", ios::out);

    for(int i=0; i < nSimulations; i++){
        nCycles = 5000 + 1000*i;

        int n = (nCycles*nParticles);
        mat positions = zeros(n, nDimensions);
        vec energy_single = zeros(n);
        vec energySquared_single = zeros(n);

        MonteCarloIntegration(nCycles, positions, energy_single, energySquared_single, variance, averange_r_ij, time);

        double energy_mean = sum(energy_single)/n;

        cout << nCycles << " " << variance << " "  << energy_mean << endl;
        outfile << nCycles << " " << variance << " "  << energy_mean << endl;
    }
    outfile.close();
}


void VMCSolver::InvestigateCPUtime(){
    int nSimulations = 10;
    int nCycles;

    double variance;
    double averange_r_ij;
    double time;

    mat time_values = zeros(nSimulations, 5);

    fstream outfile;
    outfile.open("cpu_time.dat", ios::out);


    int counter = 1;
    for(int i=1; i<=2; i++){
        for(int j=1; j<=2; j++){

            wavefunc_selection = i;
            energySolver_selection = j;

            for(int k=0; k < nSimulations; k++){
                nCycles = 500000 + 1000000*k;

                int n = (nCycles*nParticles);
                mat positions = zeros(n, nDimensions);
                vec energy_single = zeros(n);
                vec energySquared_single = zeros(n);

                MonteCarloIntegration(nCycles, positions, energy_single, energySquared_single, variance, averange_r_ij, time);

                time_values(k, 0) = nCycles;
                time_values(k, counter) = time;
            }
            counter++;
        }
    }
    outfile << time_values << endl;
    outfile.close();
}



void VMCSolver::InvestigateTimestep(){
    int nSimulations = 10;

    double variance;
    double averange_r_ij;
    double time;

    fstream outfile;
    outfile.open("timestep_dependence2.dat", ios::out);

    nCycles = 10000000;

    int counter = 0;
    for(int i=0; i < nSimulations; i++){
        timestep = 0.001 + 0.001*i;

        int n = (nCycles*nParticles);
        mat positions = zeros(n, nDimensions);
        vec energy_single = zeros(n);
        vec energySquared_single = zeros(n);

        MonteCarloIntegration(nCycles, positions, energy_single, energySquared_single, variance, averange_r_ij, time);

        double energy = sum(energy_single)/n;
        cout << timestep << " " << energy << " "  << averange_r_ij << " " << time <<  " " << nCycles << endl;
        outfile << timestep << " " << energy << " " << averange_r_ij << " " << time <<  " " << nCycles << endl;
        cout << counter << endl;
        counter++;

    }
    outfile.close();
}


void VMCSolver::BlockingFunc(){
    int nCycles = 1000000;
    int n = (nCycles*nParticles);
    mat positions = zeros(n, nDimensions);
    vec energy_single = zeros(n);
    vec energySquared_single = zeros(n);
    double variance;
    double averange_r_ij;
    double time;

    MonteCarloIntegration(nCycles, positions, energy_single, energySquared_single, variance, averange_r_ij, time);

    fstream outfile;
    outfile.open("Blocking_data_ber_imp.dat", ios::out);

    for(int i=0; i<n; i++){
        outfile << energy_single(i) << " " << energySquared_single(i) << endl;
    }
    outfile.close();
}





void VMCSolver::OnebodyDensity_ChargeDensity(){

    nCycles = 50000;
    int n = (nCycles*nParticles);
    mat positions = zeros(n*nParticles+nParticles , nDimensions);
    vec energy_single = zeros(n);
    vec energySquared_single = zeros(n);
    double variance;
    double averange_r_ij;
    double time;

    MonteCarloIntegration(nCycles, positions, energy_single, energySquared_single, variance, averange_r_ij, time);

    double energy = sum(energy_single)/n;

    fstream outfile;
    outfile.open("OnebodyDensity_ChargeDensity_beryllium_simple_imp.dat", ios::out);

    outfile << positions << endl;

    outfile.close();
}








