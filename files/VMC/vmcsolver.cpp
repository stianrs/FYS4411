#include "vmcsolver.h"
#include "lib.h"

#include <armadillo>
#include <iostream>
#include <time.h>
#include <fstream>

using namespace arma;
using namespace std;

VMCSolver::VMCSolver() :
    nDimensions(3),
    charge(2),
    Z(2),
    stepLength(1.0),
    nParticles(2),
    h(0.001),
    h2(1000000),
    idum(-1),
    alpha(1.714),
    beta(0.5714),
    nCycles(1000000),
    wavefunc_selection(2),
    energySolver_selection(2)
{
}

void VMCSolver::runMonteCarloIntegration()
{
    rOld = zeros<mat>(nParticles, nDimensions);
    rNew = zeros<mat>(nParticles, nDimensions);

    double waveFunctionOld = 0;
    double waveFunctionNew = 0;

    double energySum = 0;
    double energySquaredSum = 0;

    double r12;
    double r12_sum = 0.0;
    int r12_counter = 0;

    double deltaE;

    stepLength = InvestigateOptimalStep();

    // initial trial positions
    for(int i = 0; i < nParticles; i++) {
        for(int j = 0; j < nDimensions; j++) {
            rOld(i,j) = stepLength * (ran2(&idum) - 0.5);
        }
    }
    rNew = rOld;

    r12 = r12_func(rNew);
    r12_sum += r12;
    r12_counter += 1;

    // Start clock to compute spent time for Monte Carlo simulation
    double time;
    clock_t start, finish;
    start = clock();

    // loop over Monte Carlo cycles
    for(int cycle = 0; cycle < nCycles; cycle++) {

        // Store the current value of the wave function
        waveFunctionOld = waveFunction(rOld, wavefunc_selection);

        // New position to test
        for(int i = 0; i < nParticles; i++) {
            for(int j = 0; j < nDimensions; j++) {
                rNew(i,j) = rOld(i,j) + stepLength*(ran2(&idum) - 0.5);
            }

            // Recalculate the value of the wave function
            waveFunctionNew = waveFunction(rNew, wavefunc_selection);

            // Compute distance between electrons
            r12 = r12_func(rNew);
            r12_sum += r12;
            r12_counter += 1;

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
            deltaE = localEnergy(rNew, energySolver_selection, wavefunc_selection);
            energySum += deltaE;
            energySquaredSum += deltaE*deltaE;
        }
    }

    // Stop the clock and estimate the spent time
    finish = clock();
    time = ((finish - start)/((double) CLOCKS_PER_SEC));

    double energy = energySum/(nCycles * nParticles);
    double energySquared = energySquaredSum/(nCycles * nParticles);
    cout << "Energy: " << energy << " Energy (squared sum): " << energySquared << endl;
    cout << "Averange distance r12: " << r12_sum/r12_counter << endl;
    cout << "Time consumption for " << nCycles << " Monte Carlo samples: " << time << " sec" << endl;
}


double VMCSolver::localEnergy(const mat &r, int &energySolver_selection, int &wavefunc_selection)
{

    if (energySolver_selection == 1){

        mat rPlus = zeros<mat>(nParticles, nDimensions);
        mat rMinus = zeros<mat>(nParticles, nDimensions);

        rPlus = rMinus = r;

        double waveFunctionMinus = 0;
        double waveFunctionPlus = 0;

        double waveFunctionCurrent = waveFunction(r, wavefunc_selection);

        // Kinetic energy

        double kineticEnergy = 0;
        for(int i = 0; i < nParticles; i++) {
            for(int j = 0; j < nDimensions; j++) {
                rPlus(i,j) += h;
                rMinus(i,j) -= h;
                waveFunctionMinus = waveFunction(rMinus, wavefunc_selection);
                waveFunctionPlus = waveFunction(rPlus, wavefunc_selection);
                kineticEnergy -= (waveFunctionMinus + waveFunctionPlus - 2 * waveFunctionCurrent);
                rPlus(i,j) = r(i,j);
                rMinus(i,j) = r(i,j);
            }
        }
        kineticEnergy = 0.5 * h2 * kineticEnergy / waveFunctionCurrent;

        // Potential energy
        double potentialEnergy = 0;
        double rSingleParticle = 0;
        for(int i = 0; i < nParticles; i++) {
            rSingleParticle = 0;
            for(int j = 0; j < nDimensions; j++) {
                rSingleParticle += r(i,j)*r(i,j);
            }
            potentialEnergy -= charge / sqrt(rSingleParticle);
        }
        // Contribution from electron-electron potential
        double r12 = 0;
        for(int i = 0; i < nParticles; i++) {
            for(int j = i + 1; j < nParticles; j++) {
                r12 = 0;
                for(int k = 0; k < nDimensions; k++) {
                    r12 += (r(i,k) - r(j,k)) * (r(i,k) - r(j,k));
                }
                potentialEnergy += 1 / sqrt(r12);
            }
        }

        return kineticEnergy + potentialEnergy;
    }

    else {

        double r1;
        double r2;
        double r12;
        double r1_sum = 0;
        double r2_sum = 0;
        double r1_vec_r2_vec = 0;

        double EL1 = 0;
        double EL2 = 0;
        double compact_fraction;

        r12 = r12_func(r);

        for(int i = 0; i < nDimensions; i++){
            r1_sum += r(0, i)*r(0, i);
            r2_sum += r(1, i)*r(1, i);
            r1_vec_r2_vec += r(0, i)*r(1, i);
        }
        r1 = sqrt(r1_sum);
        r2 = sqrt(r2_sum);

        EL1 = (alpha - Z)*((1.0/r1) + (1.0/r2)) + (1.0/r12) - (alpha*alpha);

        if (wavefunc_selection == 1){
            return EL1;
        }

        else {
            compact_fraction = 1.0/(2*(1.0 + beta*r12)*(1.0 + beta*r12));
            EL2 = EL1 + compact_fraction*((alpha*(r1 + r2)/r12)*(1.0 - (r1_vec_r2_vec)/(r1*r2)) - compact_fraction - 2.0/r12 + (2.0*beta)/(1.0 + beta*r12));
            return EL2;
        }
    }
}


double VMCSolver::waveFunction(const mat &r, int &wavefunc_selection)
{

    if (wavefunc_selection == 1){
        double argument = 0;
        for(int i = 0; i < nParticles; i++) {
            double rSingleParticle = 0;
            for(int j = 0; j < nDimensions; j++) {
                rSingleParticle += r(i,j) * r(i,j);
            }
            argument += sqrt(rSingleParticle);
        }
        return exp(-argument * alpha);
    }

    else {
        double r12;
        r12 = r12_func(r);

        double argument = 0;
        for(int i = 0; i < nParticles; i++) {
            double rSingleParticle = 0;
            for(int j = 0; j < nDimensions; j++) {
                rSingleParticle += r(i,j) * r(i,j);
            }
            argument += sqrt(rSingleParticle);
        }
        return exp(-argument * alpha)*exp(r12/(2*(1.0 + beta*r12)));
    }
}

double VMCSolver::r12_func(const mat &r){
    double r12 = 0;
    for(int i = 0; i < nParticles; i++) {
        for(int j = i + 1; j < nParticles; j++) {
            r12 = 0;
            for(int k = 0; k < nDimensions; k++) {
                r12 += (r(i,k) - r(j,k)) * (r(i,k) - r(j,k));
            }
        }
    }
    return r12;
}





double VMCSolver::InvestigateOptimalStep()
{
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
    double maxStepLength = 3.0;
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
            waveFunctionOld = waveFunction(rOld, wavefunc_selection);

            // New position to test
            for(int i = 0; i < nParticles; i++) {
                for(int j = 0; j < nDimensions; j++) {
                    rNew(i,j) = rOld(i,j) + stepLength*(ran2(&idum) - 0.5);
                }

                // Recalculate the value of the wave function
                waveFunctionNew = waveFunction(rNew, wavefunc_selection);

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
                deltaE = localEnergy(rNew, energySolver_selection, wavefunc_selection);
                energySum += deltaE;
            }
        }

        ratioTrial = acceptCounter/counter;
        if (abs(ratioTrial-0.5) < abs(ratio-0.5)){
            ratio = ratioTrial;
            optimalStep = stepLength;
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
            waveFunctionOld = waveFunction(rOld, wavefunc_selection);

            // New position to test
            for(int i = 0; i < nParticles; i++) {
                for(int j = 0; j < nDimensions; j++) {
                    rNew(i,j) = rOld(i,j) + stepLength*(ran2(&idum) - 0.5);
                }

                // Recalculate the value of the wave function
                waveFunctionNew = waveFunction(rNew, wavefunc_selection);

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
                deltaE = localEnergy(rNew, energySolver_selection, wavefunc_selection);
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




void VMCSolver::InvestigateOptimalBeta(){
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
    outfile.open("Beta_Energy.dat", ios::out);

    for(int betaCounter = 0; betaCounter < nPoints; betaCounter++){
        resolution = 3.0/nPoints;
        beta = resolution*betaCounter;

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
            waveFunctionOld = waveFunction(rOld, wavefunc_selection);

            // New position to test
            for(int i = 0; i < nParticles; i++) {
                for(int j = 0; j < nDimensions; j++) {
                    rNew(i,j) = rOld(i,j) + stepLength*(ran2(&idum) - 0.5);
                }

                // Recalculate the value of the wave function
                waveFunctionNew = waveFunction(rNew, wavefunc_selection);

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
                deltaE = localEnergy(rNew, energySolver_selection, wavefunc_selection);
                energySum += deltaE;
                energySquaredSum += deltaE*deltaE;
            }
        }

        double energy = energySum/(nCycles * nParticles);
        cout << "Beta: " << beta<< "  Energy: " << energy << endl;
        outfile << beta << " " << energy << endl;

        energySum = 0.0;
        energy = 0.0;
    }
    outfile.close();
}



double VMCSolver::InvestigateOptimalParameters(){
    rOld = zeros<mat>(nParticles, nDimensions);
    rNew = zeros<mat>(nParticles, nDimensions);

    double waveFunctionOld = 0;
    double waveFunctionNew = 0;

    double energySum = 0;
    double energySquaredSum = 0;

    double deltaE;

    int nPoints = 100;
    double resolution;

    fstream outfile;
    outfile.open("Parameter_Energy2.dat", ios::out);

    for(int alphaCounter = 1; alphaCounter < nPoints; alphaCounter++){
        resolution = 3.0/nPoints;
        alpha = resolution + resolution*alphaCounter;

        for(int betaCounter = 1; betaCounter < nPoints; betaCounter++){
            resolution = 3.0/nPoints;
            beta = resolution + resolution*betaCounter;

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
                waveFunctionOld = waveFunction(rOld, wavefunc_selection);

                // New position to test
                for(int i = 0; i < nParticles; i++) {
                    for(int j = 0; j < nDimensions; j++) {
                        rNew(i,j) = rOld(i,j) + stepLength*(ran2(&idum) - 0.5);
                    }

                    // Recalculate the value of the wave function
                    waveFunctionNew = waveFunction(rNew, wavefunc_selection);

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
                    deltaE = localEnergy(rNew, energySolver_selection, wavefunc_selection);
                    energySum += deltaE;
                    energySquaredSum += deltaE*deltaE;
                }
            }

            double energy = energySum/(nCycles * nParticles);
            cout << "Alpha: " << alpha << " Beta: " << beta << " Energy: " << energy << endl;
            outfile << alpha << " " << beta << " " << energy << endl;

            energySum = 0.0;
            energy = 0.0;
        }
    }
    outfile.close();
}















