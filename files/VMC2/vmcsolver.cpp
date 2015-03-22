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
    AtomType("beryllium"),
    nDimensions(3),

    numerical_energySolver(true), // set true to solve integral numerical

    deactivate_JastrowFactor(true), // set false to activate importance sampling
    deactivate_ImportanceSampling(false), // set false to activate importance sampling
    save_positions(false), // set true to save all intermediate postitions in an MC simulation

    timestep(0.002), // timestep used in importance sampling
    D(0.5), // constant used in importance sampling
    stepLength(1.0), // set to optimal value in MC function

    h(0.001), // step used in numerical integration
    h2(1000000), // 1/h^2 used in numerical integration
    idum(time(0)) // random number generator, seed=time(0) for random seed
{    
    r_distance = zeros(nParticles, nParticles); // distance between electrons
    r_radius = zeros(nParticles); // distance between nucleus and electrons
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
        stepLength = 1.4;
    }
    else if(AtomType == "beryllium"){
        charge = 4;
        nParticles = 4;
        alpha = 3.9;
        beta = 0.1;
        stepLength = 1.4;
    }
}


// The Monte Carlo solver both with and without importance sampling
void VMCSolver::MonteCarloIntegration(int nCycles, fstream &outfile)
{
    SetParametersAtomType(AtomType);

    // fill spin matrix needed if we simulate atoms with more than 2 electrons
    fill_a_matrix();

    int n = nCycles*nParticles;

    rOld = zeros<mat>(nParticles, nDimensions);
    rNew = zeros<mat>(nParticles, nDimensions);

    energy_single = zeros<vec>(n);;
    energySquared_single = zeros<vec>(n);

    D_down_old = zeros<mat>(nParticles/2, nParticles/2);
    D_down_new = zeros<mat>(nParticles/2, nParticles/2);
    D_up_old = zeros<mat>(nParticles/2, nParticles/2);
    D_up_new = zeros<mat>(nParticles/2, nParticles/2);

    QForceOld = zeros<mat>(nParticles, nDimensions);
    QForceNew = zeros<mat>(nParticles, nDimensions);

    SlaterGradientsOld = zeros<mat>(nParticles, nDimensions);
    SlaterGradientsNew = zeros<mat>(nParticles, nDimensions);

    energy_single = zeros<vec>(n);;
    energySquared_single = zeros<vec>(n);


    if(deactivate_ImportanceSampling){
        cout << "Without importance sampling" << endl;

        double deltaE;
        double energySum = 0;
        double energySquaredSum = 0;

        double r_ij_sum = 0.0;
        int r_ij_counter = 0;

        int counter = 0;
        int counter2 = 0;
        double acceptCounter = 0;

        //stepLength = InvestigateOptimalStep();

        // initial trial positions
        for(int i = 0; i < nParticles; i++) {
            for(int j = 0; j < nDimensions; j++) {
                rOld(i,j) = stepLength * (ran2(&idum) - 0.5);
            }
        }
        rNew = rOld;

        // calculate Slater determinant for initial positions
        SlaterDeterminant(rNew);

        D_up_old = D_up_new;
        D_down_old = D_down_new;

        compute_R_sd(0);

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

            // New position to test
            for(int i = 0; i < nParticles; i++) {
                for(int j = 0; j < nDimensions; j++) {
                    rNew(i,j) = rOld(i,j) + stepLength*(ran2(&idum) - 0.5);
                }

                // Recalculate the Slater determinant
                SlaterDeterminant(rNew);

                compute_R_sd(i);
                R_c = 1; // this has to be calculated
                R = R_sd*R_c;

                cout << R << endl; // ????????????? get problems here

                // Check for step acceptance (if yes, update position, if no, reset position)
                if(ran2(&idum) <= R*R) {
                    for(int j = 0; j < nDimensions; j++) {
                        rOld(i,j) = rNew(i,j);

                        D_down_old = D_down_new;
                        D_up_old = D_up_new;

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
        cout << "With importance sampling" << endl;

        double deltaE;
        double energySum = 0;
        double energySquaredSum = 0;

        double r_ij_sum = 0.0;
        int r_ij_counter = 0;

        int counter = 0;
        int counter2 = 0;
        double acceptCounter = 0;

        // initial trial positions
        for(int i = 0; i < nParticles; i++){
            for(int j = 0; j < nDimensions; j++) {
                rOld(i,j) = GaussianDeviate(&idum)*sqrt(timestep);
            }
        }
        rNew = rOld;

        SlaterDeterminant(rNew);

        D_up_old = D_up_new;
        D_down_old = D_down_new;

        compute_R_sd(0);

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

            QuantumForce(rOld, QForceOld);

            // New position to test
            for(int i = 0; i < nParticles; i++) {
                for(int j = 0; j < nDimensions; j++) {
                    rNew(i,j) = rOld(i,j) + GaussianDeviate(&idum)*sqrt(timestep)+QForceOld(i,j)*timestep*D;
                }

                // Recalculate the value of the wave function and the quantum force
                SlaterDeterminant(rNew);
                QuantumForce(rNew,QForceNew);

                //  we compute the log of the ratio of the greens functions to be used in the
                //  Metropolis-Hastings algorithm
                GreensFunction = 0.0;
                for (int j=0; j < nDimensions; j++) {
                    GreensFunction += 0.5*(QForceOld(i,j)+QForceNew(i,j))*
                            (D*timestep*0.5*(QForceOld(i,j)-QForceNew(i,j))-rNew(i,j)+rOld(i,j));
                }
                GreensFunction = exp(GreensFunction);

                compute_R_sd(i);
                R_c = 1; // this has to be calculated
                R = R_sd*R_c;

                // The Metropolis test is performed by moving one particle at the time
                if(ran2(&idum) <= GreensFunction*R*R) {
                    for(int j = 0; j < nDimensions; j++) {
                        rOld(i,j) = rNew(i,j);
                        QForceOld(i,j) = QForceNew(i,j);

                        D_down_old = D_down_new;
                        D_up_old = D_up_new;
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

        // Kinetic energy
        double kineticEnergy = Slater_second_derivative();

        kineticEnergy = -0.5*kineticEnergy;

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
        r_radius = radius;
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


// set up spinns and compute the a-matrix
void VMCSolver::fill_a_matrix(){
    vec spin = zeros(nParticles);
    for(int i=0; i<nParticles; i++){
        if(i < nParticles/2){
            spin(i) = 1;
        }
    }
    a_matrix = zeros(nParticles, nParticles);
    double  a;
    for(int i=0; i < nParticles; i++){
        for(int j=0; j < nParticles; j++){
            if(spin(i) == spin(j)){
                a = 0.25;
            }
            else{
                a = 0.5;
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


// 1s hydrogenic orbital
double VMCSolver::psi1s(double &radius){
    double psi1s;
    psi1s = exp(-alpha*radius);
    return psi1s;
}

// 2s hydrogenic orbital
double VMCSolver::psi2s(double &radius){
    double psi2s;
    psi2s = (1.0 - alpha*radius/2.0)*exp(-alpha*radius/2.0);
    return psi2s;
}

// 2px hydrogenic orbital
double VMCSolver::psi2px(double &x, double &radius){
    double psi2px;
    psi2px = x*exp(-alpha*radius/2.0);
    return psi2px;
}

// 2py hydrogenic orbital
double VMCSolver::psi2py(double &y, double &radius){
    double psi2py;
    psi2py = y*exp(-alpha*radius/2.0);
    return psi2py;
}

// 2pz hydrogenic orbital
double VMCSolver::psi2pz(double &z, double &radius){
    double psi2pz;
    psi2pz = z*exp(-alpha*radius/2.0);
    return psi2pz;
}


double VMCSolver::SlaterPsi(const mat &positions, int i, int j){

    double radius;
    double x, y, z;

    r_func(positions);
    radius = r_radius(i);

    if(j == 0){
        return psi1s(radius);
    }
    else if(j == 1){
        return psi2s(radius);
    }
    else if(j == 2){
        x = positions(i, 0);
        return psi2px(x, radius);
    }
    else if(j == 3){
        y = positions(i, 1);
        return psi2py(y, radius);
    }
    else if(j == 4){
        z = positions(i, 2);
        return psi2pz(z, radius);
    }
    else{
        return 0;
    }
}


// compute the Slater determinant
void VMCSolver::SlaterDeterminant(const mat &positions){
    mat D_up = zeros(nParticles/2, nParticles/2);
    mat D_down = zeros(nParticles/2, nParticles/2);

    // compute spinn up part and spin down part
    for(int j=0; j<nParticles/2; j++){
        for(int i=0; i<nParticles/2; i++){
            D_up(i,j) = SlaterPsi(positions, i, j);
            D_down(i,j) = SlaterPsi(positions, i+nParticles/2, j);
        }
    }
    D_up_new = inv(D_up);
    D_down_new = inv(D_down);
}



// compute the R_sd ratio
double VMCSolver::compute_R_sd(int k){

    R_sd = 0.0;

    if (k < nParticles/2){
        for(int i=0; i<nParticles/2; i++){
            for(int j=0; j<nParticles/2; j++){
                R_sd += SlaterPsi(rNew, i, j)*D_up_old(j, i);
            }
        }
    }
    else{
        for(int i=0; i<nParticles/2; i++){
            for(int j=0; j<nParticles/2; j++){
                R_sd += SlaterPsi(rNew, i+nParticles/2, j)*D_down_old(j, i);
            }
        }
    }
    return R_sd;
}



// compute the quantum force used in importance sampling
void VMCSolver::QuantumForce(const mat &r, mat &F)
{
    for(int i = 0; i < nParticles; i++){
        Slater_first_derivative(i);
        for(int j = 0; j < nDimensions; j++) {
            F(i,j) =  2.0*SlaterGradientsNew(i,j);
        }
    }
}



// compute slater first derivative
void VMCSolver::Slater_first_derivative(int i){
    if(i < nParticles/2){
        for(int k=0; k<nDimensions; k++){
            double derivative_up = 0.0;
            for(int j=0; j<nParticles/2; j++){
                derivative_up += Psi_first_derivative(rNew, i, j, k)*D_up_old(j, i);
            }
            SlaterGradientsNew(i, k) = (1.0/R_sd)*derivative_up;
        }
    }
    else{
        for(int k=0; k<nDimensions; k++){
            double derivative_down = 0.0;
            for(int j=0; j<nParticles/2; j++){
                derivative_down += Psi_first_derivative(rNew, i, j, k)*D_down_old(j, i-nParticles/2);
            }
            SlaterGradientsNew(i, k) = (1.0/R_sd)*derivative_down;
        }
    }
}


// compute slater second derivative
double VMCSolver::Slater_second_derivative(){
    double derivative_up = 0.0;
    double derivative_down = 0.0;

    for(int i=0; i<nParticles/2; i++){
        for(int j=0; j<nParticles/2; j++){
            derivative_up += Psi_second_derivative(rNew, i, j)*D_up_old(j, i);
            derivative_down += Psi_second_derivative(rNew, i+nParticles/2, j)*D_down_old(j, i);
        }
    }
    double derivative_sum = derivative_up + derivative_down;
    return derivative_sum;
}


// Gradient of orbitals used in quantum force
double VMCSolver::Psi_first_derivative(const mat &positions, int i, int j, int k){
    // add new expression that takes in k for selceting x, y z
    double r, coor;
    double x, y, z;

    r_func(positions);
    r = r_radius(i);

    coor = rNew(i, k);

    x = rNew(i, 0);
    y = rNew(i, 1);
    z = rNew(i, 2);

    if(j == 0){
        return -alpha*coor*exp(-alpha*r)/r;
    }
    else if(j == 1){
        return 0.25*alpha*coor*(alpha*r - 4.0)*exp(-0.5*alpha*r)/r;
    }


    else if(j == 2){
        if(k==0){
            return -(0.5*alpha*x*x - r)*exp(-0.5*alpha*r)/r;
        }
        else if(k==1){
            return -0.5*alpha*x*y*exp(-0.5*alpha*r)/r;
        }
        else{
            return -0.5*alpha*x*z*exp(-0.5*alpha*r)/r;
        }
    }


    else if(j == 3){
        if(k==0){
            return -0.5*alpha*x*y*exp(-0.5*alpha*r)/r;
        }
        else if(k==1){
            return -(0.5*alpha*y*y - r)*exp(-0.5*alpha*r)/r;
        }
        else{
            return -0.5*alpha*y*z*exp(-0.5*alpha*r)/r;
        }
    }


    else if(j == 4){
        if(k==0){
            return -0.5*alpha*x*z*exp(-0.5*alpha*r)/r;
        }
        else if(k==1){
            return -0.5*alpha*y*z*exp(-0.5*alpha*r)/r;
        }
        else{
            return -(0.5*alpha*z*z - r)*exp(-0.5*alpha*r)/r;
        }
    }
    else{
        return 0;
    }
}


// Laplacian of orbitals used in kinetic energy
double VMCSolver::Psi_second_derivative(const mat &positions, int i, int j){
    double r;
    double x, y, z;

    r_func(positions);

    r = r_radius(i);
    x = rNew(i, 0);
    y = rNew(i, 1);
    z = rNew(i, 2);

    double x2 = x*x;
    double y2 = y*y;
    double z2 = z*z;
    double alpha2 = alpha*alpha;

    if(j == 0){
        return alpha*(alpha*r - 2)*exp(-alpha*r)/r;
    }
    else if(j == 1){
        return -0.015625*alpha*(0.125*alpha2*x2 + 0.125*alpha2*y2 + 0.125*alpha2*z2 - 1.25*alpha*r + 2.0)*exp(-0.5*alpha*r)/r;
    }
    else if(j == 2){
        return 0.0625*alpha2*x*(0.25*alpha*r - 2.0)*exp(-0.5*alpha*r)/r;
    }
    else if(j == 3){
        return 0.0625*alpha2*y*(0.25*alpha*r - 2.0)*exp(-0.5*alpha*r)/r;
    }
    else if(j == 4){
        return 0.0625*alpha2*z*(0.25*alpha*r - 2.0)*exp(-0.5*alpha*r)/r;
    }
    else{
        return 0;
    }
}












