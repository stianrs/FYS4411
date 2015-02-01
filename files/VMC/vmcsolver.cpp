#include "vmcsolver.h"
#include "lib.h"

#include <armadillo>
#include <iostream>
#include <time.h>

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
    alpha(0.5*charge),
    beta(1.0),
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
    cout << "Time consumption for " << nCycles << "Monte Carlo samples: " << time << " sec" << endl;
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
















