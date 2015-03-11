/*
This is the program with the MC solver both with and without importrance sampling
*/

#include "vmcsolver.h"
#include "lib.h"
#include "investigate.h"

#include <armadillo>
#include <iostream>
#include <time.h>
#include <fstream>

using namespace arma;
using namespace std;

VMCSolver::VMCSolver():
    AtomType("helium"),
    nDimensions(3),

    numerical_energySolver(false), // set true to solve integral numerical

    deactivate_JastrowFactor(false), // set false to activate importance sampling
    deactivate_ImportanceSampling(true), // set false to activate importance sampling
    save_positions(false), // set true to save all intermediate postitions in an MC simulation

    timestep(0.002), // timestep used in importance sampling
    D(0.5), // constant used in importance sampling
    stepLength(1.0), // set to optimal value in MC function

    h(0.001), // step used in numerical integration
    h2(1000000) // 1/h^2 used in numerical integration

{
    r_distance = zeros(nParticles, nParticles); // distance between electrons
    r_centre = zeros(nParticles); // distance between nucleus and electrons
}

// function to run MC simualtions form main.cpp
void VMCSolver::runMonteCarloIntegration(int nCycles)
{
    fstream outfile;
    MonteCarloIntegration(nCycles, outfile);
}


void VMCSolver::SetParametersAtomType(string AtomType){
    if (AtomType == "helium"){
        charge = 2;
        nParticles = 2;
        alpha = 1.85;
        beta = 0.35;
    }
    else if(AtomType == "beryllium"){
        charge = 4;
        nParticles = 4;
        alpha = 3.9;
        beta = 0.1;
    }
}


// The Monte Carlo solver both with and without importance sampling
void VMCSolver::MonteCarloIntegration(int nCycles, fstream &outfile)
{
    SetParametersAtomType(AtomType);

    // fill spin matrix needed if we simulate atoms with more than 2 electrons
    fill_a_matrix();

    // random seed in random number generator
    time_t idum = time(0);

    if(deactivate_ImportanceSampling){
        cout << "Without importance sampling" << "  Wavefunc_selection: " << deactivate_JastrowFactor << "  energySolver_selection: " << numerical_energySolver << endl;

        int n = nCycles*nParticles;

        rOld = zeros<mat>(nParticles, nDimensions);
        rNew = zeros<mat>(nParticles, nDimensions);

        energy_single = zeros<vec>(n);;
        energySquared_single = zeros<vec>(n);


        double waveFunctionOld = 0;
        double waveFunctionNew = 0;

        double deltaE;
        double energySum = 0;
        double energySquaredSum = 0;

        double r_ij_sum = 0.0;
        int r_ij_counter = 0;

        int counter = 0;
        int counter2 = 0;
        double acceptCounter = 0;

        stepLength = InvestigateOptimalStep();

        // initial trial positions
        for(int i = 0; i < nParticles; i++) {
            for(int j = 0; j < nDimensions; j++) {
                rOld(i,j) = stepLength * (ran2(&idum) - 0.5);
            }
        }
        rNew = rOld;


        // store initial position
        if (save_positions){
            save_positions_func(rNew, outfile);
        }

        // calculate distance between electrons
        r_func(rNew);
        int div = nParticles*nParticles - nParticles;
        r_ij_sum += sum(sum(r_distance))/div;
        r_ij_counter += 1;

        // Start clock to compute spent time for Monte Carlo simulation
        clock_t start, finish;
        start = clock();

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
                    acceptCounter += 1;
                    counter2 += 1;
                } else {
                    for(int j = 0; j < nDimensions; j++) {
                        rNew(i,j) = rOld(i,j);
                    }
                    counter2 += 1;
                }

                // Compute new distance between electrons
                r_func(rNew);
                r_ij_sum += sum(sum(r_distance))/div;
                r_ij_counter += 1;


                // update energies
                deltaE = localEnergy(rNew);
                energySum += deltaE;
                energySquaredSum += deltaE*deltaE;

                energy_single(counter) = deltaE;
                energySquared_single(counter) = deltaE*deltaE;
                counter += 1;

                // store all intermediate positions
                if(save_positions){
                    save_positions_func(rNew, outfile);
                }
            }
        }

        double ratioTrial = acceptCounter/counter2;
        cout << "Acceptance ratio: " << ratioTrial << endl;

        // Stop the clock and estimate the spent time
        finish = clock();
        cpu_time = ((finish - start)/((double) CLOCKS_PER_SEC));

        double energy_mean = energySum/n;
        double energySquared_mean = energySquaredSum/n;

        variance = (energySquared_mean - (energy_mean*energy_mean))/n;
        averange_r_ij = r_ij_sum/r_ij_counter;

        cout << "Energy: " << energy_mean << " Variance: " << variance <<  " Averange distance r_ij: " << r_ij_sum/r_ij_counter << endl;
        cout << "Time consumption for " << nCycles << " Monte Carlo samples: " << cpu_time << " sec" << endl;
    }

    else{
        cout << "With importance sampling" << "  Wavefunc_selection: " << deactivate_JastrowFactor << "  energySolver_selection: " << numerical_energySolver << endl;

        int n = nCycles*nParticles;

        rOld = zeros<mat>(nParticles, nDimensions);
        rNew = zeros<mat>(nParticles, nDimensions);
        QForceOld = zeros<mat>(nParticles, nDimensions);
        QForceNew = zeros<mat>(nParticles, nDimensions);

        energy_single = zeros<vec>(n);;
        energySquared_single = zeros<vec>(n);

        double waveFunctionOld = 0;
        double waveFunctionNew = 0;

        double deltaE;
        double energySum = 0;
        double energySquaredSum = 0;

        double r_ij_sum = 0.0;
        int r_ij_counter = 0;

        int counter = 0;
        int counter2 = 0;
        double acceptCounter = 0;

        vec positions = zeros(nParticles*nDimensions);


        // initial trial positions
        for(int i = 0; i < nParticles; i++){
            for(int j = 0; j < nDimensions; j++) {
                rOld(i,j) = GaussianDeviate(&idum)*sqrt(timestep);
            }
        }
        rNew = rOld;

        // store initial position
        if (save_positions){
            save_positions_func(rNew, outfile);
        }

        // calculate distance between electrons
        r_func(rNew);
        int div = nParticles*nParticles - nParticles;
        r_ij_sum += sum(sum(r_distance))/div;
        r_ij_counter += 1;

        // Start clock to compute spent time for Monte Carlo simulation
        clock_t start, finish;
        start = clock();

        // loop over Monte Carlo cycles
        for(int cycle = 0; cycle < nCycles; cycle++) {

            // Store the current value of the wave function
            waveFunctionOld = waveFunction(rOld);
            QuantumForce(rOld, QForceOld);
            QForceOld = QForceOld*h/waveFunctionOld;

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
                waveFunctionNew = waveFunction(rNew);
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
                    acceptCounter += 1;
                    counter2 += 1;
                } else {
                    for(int j = 0; j < nDimensions; j++) {
                        rNew(i,j) = rOld(i,j);
                        QForceNew(i,j) = QForceOld(i,j);
                    }
                    counter2 += 1;
                }

                // Compute distance between electrons
                r_func(rNew);
                r_ij_sum += sum(sum(r_distance))/div;
                r_ij_counter += 1;

                // update energies
                deltaE = localEnergy(rNew);
                energySum += deltaE;
                energySquaredSum += deltaE*deltaE;

                energy_single(counter) = deltaE;
                energySquared_single(counter) = deltaE*deltaE;
                counter += 1;

                // store all intermediate positions
                if(save_positions){
                    save_positions_func(rNew, outfile);
                }
            }
        }

        double ratioTrial = acceptCounter/counter2;
        cout << "Acceptance ratio: " << ratioTrial << endl;

        // Stop the clock and estimate the spent time
        finish = clock();
        cpu_time = ((finish - start)/((double) CLOCKS_PER_SEC));

        double energy_mean = energySum/n;
        double energySquared_mean = energySquaredSum/n;

        variance = (energySquared_mean - (energy_mean*energy_mean))/n;
        averange_r_ij = r_ij_sum/r_ij_counter;

        cout << "Energy: " << energy_mean << " Variance: " << variance <<  " Averange distance r_ij: " << r_ij_sum/r_ij_counter << endl;
        cout << "Time consumption for " << nCycles << " Monte Carlo samples: " << cpu_time << " sec" << endl;
    }
}



// function to compute local energy both numerical and analytical (if expression is found by user)
double VMCSolver::localEnergy(const mat &r)
{

    // numerical computation of local energy
    if (numerical_energySolver){

        mat rPlus = zeros<mat>(nParticles, nDimensions);
        mat rMinus = zeros<mat>(nParticles, nDimensions);

        rPlus = rMinus = r;

        double waveFunctionMinus = 0;
        double waveFunctionPlus = 0;

        double waveFunctionCurrent = waveFunction(r);

        // Kinetic energy
        double kineticEnergy = 0;
        for(int i = 0; i < nParticles; i++) {
            for(int j = 0; j < nDimensions; j++) {
                rPlus(i,j) += h;
                rMinus(i,j) -= h;
                waveFunctionMinus = waveFunction(rMinus);
                waveFunctionPlus = waveFunction(rPlus);
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
        for(int i = 0; i < nParticles; i++) {
            for(int j = i+1; j < nParticles; j++) {
                double r_ij = 0;
                for(int k = 0; k < nDimensions; k++) {
                    r_ij += (r(i,k) - r(j,k)) * (r(i,k) - r(j,k));
                }
                potentialEnergy += 1.0/sqrt(r_ij);
            }
        }
        return kineticEnergy + potentialEnergy;
    }

    // analytical expressions for local energy
    else {

        if (AtomType == "helium"){
            double r1;
            double r2;
            double r1_sum = 0;
            double r2_sum = 0;
            double r1_vec_r2_vec = 0;

            double EL1 = 0;
            double EL2 = 0;
            double compact_fraction;

            r_func(r);
            int div = nParticles*nParticles - nParticles;
            double r12 = sum(sum(r_distance))/div;
            for(int i = 0; i < nDimensions; i++){
                r1_sum += r(0, i)*r(0, i);
                r2_sum += r(1, i)*r(1, i);
                r1_vec_r2_vec += r(0, i)*r(1, i);
            }
            r1 = sqrt(r1_sum);
            r2 = sqrt(r2_sum);

            EL1 = (alpha - charge)*((1.0/r1) + (1.0/r2)) + (1.0/r12) - (alpha*alpha);

            if (deactivate_JastrowFactor){
                return EL1;
            }

            else {
                compact_fraction = 1.0/(2*(1.0 + beta*r12)*(1.0 + beta*r12));
                EL2 = EL1 + compact_fraction*((alpha*(r1 + r2)/r12)*(1.0 - (r1_vec_r2_vec)/(r1*r2)) - compact_fraction - 2.0/r12 + (2.0*beta)/(1.0 + beta*r12));
                return EL2;
            }
        }
        else{
            cout << "You need to implement an analytical expression!" << endl;
            exit(0);
        }
    }
}


// function that compute wavefunctions given with to compute bu user
double VMCSolver::waveFunction(const mat &r)
{

    if (AtomType == "helium"){
        if (deactivate_JastrowFactor){
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

        if (deactivate_JastrowFactor == false){
            r_func(r);
            int div = nParticles*nParticles - nParticles;
            double r12 = sum(sum(r_distance))/div;

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

    else if (AtomType == "beryllium"){
        if (deactivate_JastrowFactor){
            r_func(r);

            double hydrogenic = SlaterDeterminant();
            return hydrogenic;
        }

        if (deactivate_JastrowFactor == false){
            r_func(r);

            double factor = JastrowFactor();
            double hydrogenic = SlaterDeterminant();
            return hydrogenic*factor;
        }

        else{
            exit(0);
        }
    }
}



// compute the distance between all electrons in the atom and the distance from the nucleus
void VMCSolver::r_func(const mat &positions){
    mat distance = zeros(nParticles, nParticles);
    vec radius = zeros(nParticles);
    for(int i = 0; i < nParticles; i++){
        for(int j = 0; j < i+1; j++) {
            double r_ij = 0;
            double r_position = 0;
            for(int k = 0; k < nDimensions; k++){
                r_ij += (positions(i,k) - positions(j,k)) * (positions(i,k) - positions(j,k));
                r_position += positions(i,k)*positions(i,k);
            }
            distance(i,j) = sqrt(r_ij);
            distance(j,i) = distance(i,j);
            radius(i) = sqrt(r_position);
        }
        r_distance = distance;
        r_centre = radius;
    }
}


// write positons to file
void VMCSolver::save_positions_func(const mat &r, fstream &outfile){
    int counter_pos = 0;
    for(int i=0; i < nParticles; i++){
        for(int j=0; j < nDimensions; j++){
            outfile << r(i, j) << " ";
            counter_pos++;
        }
    }
    outfile << endl;
}



// compute the quantum force used in importance sampling
void VMCSolver::QuantumForce(const mat &r, mat &QForce)
{
    mat rPlus = zeros<mat>(nParticles, nDimensions);
    mat rMinus = zeros<mat>(nParticles, nDimensions);

    rPlus = rMinus = r;

    double waveFunctionMinus = 0;
    double waveFunctionPlus = 0;

    for(int i = 0; i < nParticles; i++) {
        for(int j = 0; j < nDimensions; j++) {
            rPlus(i,j) += h;
            rMinus(i,j) -= h;
            waveFunctionMinus = waveFunction(rMinus);
            waveFunctionPlus = waveFunction(rPlus);
            QForce(i,j) =  (waveFunctionPlus-waveFunctionMinus);
            rPlus(i,j) = r(i,j);
            rMinus(i,j) = r(i,j);
        }
    }
}


// 1s hydrogenic orbital
double VMCSolver::psi1s(double &r){
    double psi1s;
    psi1s = exp(-alpha*r);
    return psi1s;
}

// 2s hydrogenic orbital
double VMCSolver::psi2s(double &r){
    double psi2s;
    psi2s = (1.0 - alpha*r/2.0)*exp(-alpha*r/2.0);
    return psi2s;
}

// 2p hydrogenic orbital
double VMCSolver::psi2p(double &r){
    double psi2p;
    psi2p = alpha*r*exp(-alpha*r/2.0);
    return psi2p;
}

// set up spinns and compute the a-matrix
void VMCSolver::fill_a_matrix(){
    vec spin = zeros(nParticles);
    for(int i=0; i<nParticles; i++){
        if(i < nParticles/2){
            spin(i) = 1;
        }
        else{
            spin(i) = 0;
        }
    }
    a_matrix = zeros(nParticles, nParticles);
    double  a;
    for(int i=0; i < nParticles; i++){
        for(int j=0; j < nParticles; j++){
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

// compute the Jastrow factor
double VMCSolver::JastrowFactor(){
    double Psi = 1.0;
    for(int j=0; j < nParticles; j++){
        for(int i=0; i < j; i++){
            Psi *= exp((a_matrix(i,j)*r_distance(i,j))/(1.0 + beta*r_distance(i,j)));
        }
    }
    return Psi;
}

// compute the Slater determinant for Beryllium
double VMCSolver::SlaterDeterminant(){
    vec argument = zeros(nParticles);

    argument = r_centre;
    // Slater determinant, no factors as they vanish in Metropolis ratio
    double wf  = (psi1s(argument(0))*psi2s(argument(1))-psi1s(argument(1))*psi2s(argument(0)))*(psi1s(argument(2))*psi2s(argument(3))-psi1s(argument(3))*psi2s(argument(2)));
    return wf;
}




