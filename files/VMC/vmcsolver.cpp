#include "vmcsolver.h"
#include "lib.h"
#include "investigate.h"

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
    alpha(1.843), //alpha(1.843),
    beta(0.347), //beta(0.347),
    nCycles(1000000),
    timestep(0.05), // IS
    D(0.5), // IS

    wavefunc_selection(1),
    energySolver_selection(2),
    activate_ImportanceSampling(1)

{
}

void VMCSolver::runMonteCarloIntegration(int nCycles)
{
    int n = (nCycles*nParticles);
    vec energy_single = zeros(n);
    vec energySquared_single = zeros(n);
    double variance;
    double averange_r12;
    double time;

    if(activate_JastrowFactor == 1){
        MonteCarloIntegration(nCycles, energy_single, energySquared_single, variance, averange_r12, time);
    }
    else{
        importanceMonteCarloIntegration(nCycles, energy_single, energySquared_single, variance, averange_r12, time);
    }
}



void VMCSolver::MonteCarloIntegration(int nCycles, vec &energy_single, vec &energySquared_single, double &variance, double &averange_r12, double &time)
{
    rOld = zeros<mat>(nParticles, nDimensions);
    rNew = zeros<mat>(nParticles, nDimensions);

    int n = nCycles*nParticles;

    double waveFunctionOld = 0;
    double waveFunctionNew = 0;

    double energySum = 0;
    double energySquaredSum = 0;

    double r12;
    double r12_sum = 0.0;
    int r12_counter = 0;

    double deltaE;
    int counter = 0;

    stepLength = 1.4;
    //stepLength = InvestigateOptimalStep();

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

            // Compute distance between electrons
            r12 = r12_func(rNew);
            r12_sum += r12;
            r12_counter += 1;


            // update energies
            deltaE = localEnergy(rNew, energySolver_selection, wavefunc_selection);
            energySum += deltaE;
            energySquaredSum += deltaE*deltaE;

            energy_single(counter) = deltaE;
            energySquared_single(counter) = deltaE*deltaE;
            counter += 1;
        }
    }

    // Stop the clock and estimate the spent time
    finish = clock();
    time = ((finish - start)/((double) CLOCKS_PER_SEC));

    double energy_mean = energySum/n;
    double energySquared_mean = energySquaredSum/n;

    variance = (energySquared_mean - (energy_mean*energy_mean))/n;
    averange_r12 = r12_sum/r12_counter;

    cout << "Variance: " << variance << " Averange distance r12: " << r12_sum/r12_counter << endl;
    cout << "Time consumption for " << nCycles << " Monte Carlo samples: " << time << " sec" << endl;
}


void VMCSolver::importanceMonteCarloIntegration(int nCycles, vec &energy_single, vec &energySquared_single, double &variance, double &averange_r12, double &time)
{
    rOld = zeros<mat>(nParticles, nDimensions);
    rNew = zeros<mat>(nParticles, nDimensions);
    QForceOld = zeros<mat>(nParticles, nDimensions);
    QForceNew = zeros<mat>(nParticles, nDimensions);

    int n = nCycles*nParticles;

    double waveFunctionOld = 0;
    double waveFunctionNew = 0;

    double energySum = 0;
    double energySquaredSum = 0;

    double r12;
    double r12_sum = 0.0;
    int r12_counter = 0;

    double deltaE;
    int counter = 0;

    stepLength = 1.4;
    //stepLength = InvestigateOptimalStep();

    // initial trial positions
    for(int i = 0; i < nParticles; i++) {
        for(int j = 0; j < nDimensions; j++) {
            rOld(i,j) = GaussianDeviate(&idum)*sqrt(timestep);
        }
    }
    rNew = rOld;

    r12 = r12_func(rNew);
    r12_sum += r12;
    r12_counter += 1;


    // Start clock to compute spent time for Monte Carlo simulation
    clock_t start, finish;
    start = clock();

    // loop over Monte Carlo cycles
    for(int cycle = 0; cycle < nCycles; cycle++) {

        // Store the current value of the wave function
        waveFunctionOld = waveFunction(rOld, wavefunc_selection);
        QuantumForce(rOld, QForceOld); QForceOld = QForceOld*h/waveFunctionOld;

        // New position to test
        for(int i = 0; i < nParticles; i++) {
            for(int j = 0; j < nDimensions; j++) {
                rNew(i,j) = rOld(i,j) + GaussianDeviate(&idum)*sqrt(timestep)+QForceOld(i,j)*timestep*D;
            }
            //  for the other particles we need to set the position to the old position since
            //  we move only one particle at the time
            for (int k = 0; k < nParticles; k++) {
                if ( k != i) {
                    for (int j=0; j < nDimensions; j++) {
                        rNew(k,j) = rOld(k,j);
                    }
                }
            }

            // Recalculate the value of the wave function and the quantum force
            waveFunctionNew = waveFunction(rNew, wavefunc_selection);
            QuantumForce(rNew,QForceNew); QForceNew*h/waveFunctionNew;

            //  we compute the log of the ratio of the greens functions to be used in the
            //  Metropolis-Hastings algorithm
            GreensFunction = 0.0;
            for (int j=0; j < nDimensions; j++) {
                GreensFunction += 0.5*(QForceOld(i,j)+QForceNew(i,j))*
                        (D*timestep*0.5*(QForceOld(i,j)-QForceNew(i,j))-rNew(i,j)+rOld(i,j));
            }
            GreensFunction = exp(GreensFunction);


            // The Metropolis test is performed by moving one particle at the time
            if(ran2(&idum) <= GreensFunction*(waveFunctionNew*waveFunctionNew) / (waveFunctionOld*waveFunctionOld)) {
                for(int j = 0; j < nDimensions; j++) {
                    rOld(i,j) = rNew(i,j);
                    QForceOld(i,j) = QForceNew(i,j);
                    waveFunctionOld = waveFunctionNew;
                }
            } else {
                for(int j = 0; j < nDimensions; j++) {
                    rNew(i,j) = rOld(i,j);
                    QForceNew(i,j) = QForceOld(i,j);
                }
            }


            // Compute distance between electrons
            r12 = r12_func(rNew);
            r12_sum += r12;
            r12_counter += 1;


            // update energies
            deltaE = localEnergy(rNew, energySolver_selection, wavefunc_selection);
            energySum += deltaE;
            energySquaredSum += deltaE*deltaE;

            energy_single(counter) = deltaE;
            energySquared_single(counter) = deltaE*deltaE;
            counter += 1;
        }
    }

    // Stop the clock and estimate the spent time
    finish = clock();
    time = ((finish - start)/((double) CLOCKS_PER_SEC));

    double energy_mean = energySum/n;
    double energySquared_mean = energySquaredSum/n;

    variance = (energySquared_mean - (energy_mean*energy_mean))/n;
    averange_r12 = r12_sum/r12_counter;

    cout << "Variance: " << variance << " Averange distance r12: " << r12_sum/r12_counter << endl;
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
    return sqrt(r12);
}




double VMCSolver::QuantumForce(const mat &r, mat QForce)
{
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
            QForce(i,j) =  (waveFunctionPlus-waveFunctionMinus);
            rPlus(i,j) = r(i,j);
            rMinus(i,j) = r(i,j);
        }
    }
}



double VMCSolver::psi1s(r){
    double psi1s;
    psi1s = exp(-alpha*r);
    return psi1s;
}

double VMCSolver::psi2s(r){
    double psi2s;
    psi2s = (1.0 - alpha*r/2.0)*exp(-alpha*r/2.0);
    return psi2s;
}



void VMCSolver::fill_a_matrix(){
    vec spin = zeros(nParticles);
    a_matrix = zeros(nParticles, nParticles);

    spin(0) = 1;
    spin(1) = 1;
    spin(2) = 0;
    spin(3) = 0;

    double  a;
    for(int i=0; i<nParticles; i++){
        for(int j=0; j<nParticles; j++){
            if(spin(i) == spin(j)){
                a = 1.0/4.0;
            }
            else{
                a = 1.0/2.0;
            }
            a_matrix(i,j) = a;
        }
    }
}



void VMCSolver::JastrowFactor(){
    double Psi;
    for(int j=0; j<nParticles; j++){
        for(int i=0; i<j; i++){
            Psi *= exp((a_matrix(i,j)*r(i,j))/(1.0 + beta*r(i,j)));
        }
    }
}







void VMCSolver::SlaterDeterminant(){
    vec argument = zeros(nParticles);

    for (i = 0; i < nParticles; i++) {
        argument[i] = 0.0;
        r_single_particle = 0;
        for (j = 0; j < dimension; j++) {
            r_single_particle  += r[i][j]*r[i][j];
        }
        argument[i] = sqrt(r_single_particle);
    }


    // Slater determinant, no factors as they vanish in Metropolis ratio
    wf  = (psi1s(argument[0])*psi2s(argument[1])
            -psi1s(argument[1])*psi2s(argument[0]))*
            (psi1s(argument[2])*psi2s(argument[3])
            -psi1s(argument[3])*psi2s(argument[2]));

}

}





