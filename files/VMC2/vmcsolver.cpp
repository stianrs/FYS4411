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
    nDimensions(2),
    
    energySelector("optimized"),
    activate_JastrowFactor(true), // set true to activate importance sampling
    save_positions(false), // set true to save all intermediate postitions in an MC simulation
    
    timestep(0.0002), // timestep used in importance sampling
    D(0.5), // constant used in importance sampling
    
    h(0.001), // step used in numerical integration
    h2(1000000), // 1/h^2 used in numerical integration
    idum(time(0)) // random number generator, seed=time(0) for random seed
    //idum(-1)
{
    r_distance = zeros(nParticles, nParticles); // distance between electrons
    r_radius = zeros(nParticles); // distance between nucleus and electrons
}

// function to run MC simualtions from main.cpp
double VMCSolver::runMonteCarloIntegration(int nCycles, int my_rank, int world_size)
{
    fstream outfile;
    MonteCarloIntegration(nCycles, outfile, my_rank, world_size);
    
    return energy_estimate;
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
    else if(AtomType == "neon"){
        charge = 10;
        nParticles = 10;
        alpha = 10.2;
        beta = 0.09;
    }
}


// The Monte Carlo solver both with and without importance sampling
void VMCSolver::MonteCarloIntegration(int nCycles, fstream &outfile, int my_rank, int world_size)
{
    // workload for different processors
    nCycles = nCycles/world_size;
    
    // Start clock to compute spent time for Monte Carlo simulation
    clock_t start, finish;
    start = clock();
    
    // Random seed between parallel processors
    this->idum += my_rank*time(0);
    
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
    
    SlaterGradientOld = zeros<mat>(nParticles, nDimensions);
    SlaterGradientNew = zeros<mat>(nParticles, nDimensions);
    
    C_old = zeros<mat>(nParticles, nParticles);
    C_new = zeros<mat>(nParticles, nParticles);
    
    JastrowDerivative = zeros<mat>(nParticles, nParticles);
    JastrowDerivativeOld = zeros<mat>(nParticles, nParticles);
    JastrowGradient = zeros<mat>(nParticles, nDimensions);
    JastrowLaplacianNew = zeros<mat>(nParticles, nParticles);
    JastrowLaplacianOld = zeros<mat>(nParticles, nParticles);
    
    energy_single = zeros<vec>(n);;
    energySquared_single = zeros<vec>(n);
    
    JastrowEnergySum = 0.0;
    CrosstermSum = 0.0;
    
    double deltaE;
    double energySum = 0;
    double energySquaredSum = 0;
    
    double r_ij_sum = 0.0;
    int r_ij_counter = 0;
    
    int counter = 0;
    double acceptCounter = 0;
        
    // initial trial positions
    for(int i = 0; i < nParticles; i++){
        for(int j = 0; j < nDimensions; j++) {
            rOld(i,j) = GaussianDeviate(&idum)*0.5;
        }
    }
    rNew = rOld;
    
    // Calculate r_distance and r_radius
    r_func(rNew);
    
    R_sd = 1.0;

    // Compute everything around Slaterdeterminant
    SlaterDeterminant(rNew);
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

    // store initial position
    if (save_positions){
        save_positions_func(rNew, outfile);
    }

    // calculate distance between electrons
    int div = nParticles*nParticles - nParticles;
    r_ij_sum += sum(sum(r_distance))/div;
    r_ij_counter += 1;

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

            SlaterGradient(i);
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

                energy_single(counter) = deltaE;
                energySquared_single(counter) = deltaE*deltaE;
                counter += 1;

                // Compute distance between electrons
                r_ij_sum += sum(sum(r_distance))/div;
                r_ij_counter += 1;

                // store all intermediate positions
                if(save_positions){
                    save_positions_func(rNew, outfile);
                }
            }

            else {
                for(int j=0; j<nDimensions; j++) {
                    rNew(i,j) = rOld(i,j);
                }

                // compute energies
                deltaE = localEnergy(rNew);
                energySum += deltaE;
                energySquaredSum += deltaE*deltaE;

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
    double ratioTrial = acceptCounter/n;
    double energy_mean = energySum/n;
    double energySquared_mean = energySquaredSum/n;

    energy_estimate = energy_mean;
    variance = (energySquared_mean - (energy_mean*energy_mean))/n;
    averange_r_ij = r_ij_sum/r_ij_counter;

    // Stop the clock and estimate the spent time
    finish = clock();
    cpu_time = ((finish - start)/((double) CLOCKS_PER_SEC));

    cout << "With importance sampling, prosessor: "  << my_rank << endl;
    cout << "Acceptance ratio: " << ratioTrial << endl;
    cout << "Energy: " << energy_mean << " Variance: " << variance <<  " Averange distance r_ij: " << r_ij_sum/r_ij_counter << endl;
    cout << "Time consumption for " << nCycles << " Monte Carlo samples: " << cpu_time << " sec" << endl;
    cout << endl;
}


// function to compute local energy both numerical and analytical (if expression is found by user)
double VMCSolver::localEnergy(const mat &r)
{
    // numerical computation of local energy
    if (energySelector == "optimized"){

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
        if (activate_JastrowFactor){
            double slaterLaplacianEnergy = SlaterLaplacian();
            double JastrowEnergy = computeJastrowEnergy();
            double kineticEnergy = -0.5*(slaterLaplacianEnergy + JastrowEnergy + 2.0*energytermSlaterJastrow);
            JastrowEnergySum += JastrowEnergy;
            CrosstermSum += 2.0*energytermSlaterJastrow;
            return kineticEnergy + potentialEnergy;
        }
        else{
            double kineticEnergy = -0.5*SlaterLaplacian();
            return kineticEnergy + potentialEnergy;
        }
    }


    else if(energySelector == "numerical"){
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

    // analytical expressions for local energy of Helium
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

            if (activate_JastrowFactor){
                compact_fraction = 1.0/(2*(1.0 + beta*r12)*(1.0 + beta*r12));
                EL2 = EL1 + compact_fraction*((alpha*(r1 + r2)/r12)*(1.0 - (r1_vec_r2_vec)/(r1*r2)) - compact_fraction - 2.0/r12 + (2.0*beta)/(1.0 + beta*r12));
                return EL2;
            }

            else {
                return EL1;
            }
        }
        else{
            cout << "You need to implement an analytical expression!" << endl;
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


// compute the Slater determinant for the first time
void VMCSolver::SlaterDeterminant(const mat &positions){
    mat D_up = zeros(nParticles/2, nParticles/2);
    mat D_down = zeros(nParticles/2, nParticles/2);

    // compute spinn up part and spin down part
    for(int j=0; j<nParticles/2; j++){
        for(int i=0; i<nParticles/2; i++){
            D_up(i,j) = SlaterPsi(positions, i, j);
            D_down(i,j) = SlaterPsi(positions, i+(nParticles/2), j);
        }
    }
    D_up_new = D_up.i();
    D_down_new = D_down.i();
}


// compute the R_sd ratio
void VMCSolver::compute_R_sd(int i){
    R_sd = 0.0;

    if(i < nParticles/2){
        for(int j=0; j<nParticles/2; j++){
            R_sd += SlaterPsi(rNew, i, j)*D_up_old(j, i);
        }
    }
    else{
        for(int j=0; j<nParticles/2; j++){
            R_sd += SlaterPsi(rNew, i, j)*D_down_old(j, i-nParticles/2);
        }
    }
}


// compute R_c ratio
void VMCSolver::compute_R_c(int k){
    double deltaU = 0.0;
    for(int i=0; i < k; i++){
        deltaU += C_new(i, k) - C_old(i, k);
    }
    for(int i=k+1; i<nParticles; i++){
        deltaU += C_new(k, i) - C_old(k, i);
    }
    R_c = exp(deltaU);
}


// compute slater first derivative
void VMCSolver::SlaterGradient(int i){
    if(i < nParticles/2){
        for(int k=0; k<nDimensions; k++){
            double derivative_up = 0.0;
            for(int j=0; j<nParticles/2; j++){
                derivative_up += Psi_first_derivative(i, j, k)*D_up_old(j, i);
            }
            SlaterGradientNew(i, k) = (1.0/R_sd)*derivative_up;
        }
    }
    else{
        for(int k=0; k<nDimensions; k++){
            double derivative_down = 0.0;
            for(int j=0; j<nParticles/2; j++){
                derivative_down += Psi_first_derivative(i, j, k)*D_down_old(j, i-nParticles/2);
            }
            SlaterGradientNew(i, k) = (1.0/R_sd)*derivative_down;
        }
    }
}


// compute slater second derivative
double VMCSolver::SlaterLaplacian(){
    double derivative_up = 0.0;
    double derivative_down = 0.0;

    for(int i=0; i<nParticles/2; i++){
        for(int j=0; j<nParticles/2; j++){
            derivative_up += Psi_second_derivative(i, j)*D_up_new(j, i);
            derivative_down += Psi_second_derivative(i+nParticles/2, j)*D_down_new(j, i);
        }
    }
    double derivative_sum = derivative_up + derivative_down;
    return derivative_sum;
}


// NB: Not used in the computations
// compute the Jastrow factor
double VMCSolver::ComputeJastrow(){
    double corr = 0.0;

    for(int k=0; k < nParticles; k++){
        for(int i=k+1; i < nParticles; i++){
            corr += a_matrix(k,i)*r_distance(k,i)/(1.0 + beta*r_distance(k,i));
        }
    }
    corr = exp(corr);
    return corr;
}


// fill Jastrow matrix for use the first time
void VMCSolver::fillJastrowMatrix(mat &CorrelationMatrix){
    for(int k=0; k < nParticles; k++){
        for(int i=k+1; i < nParticles; i++){
            CorrelationMatrix(k, i) = a_matrix(k,i)*r_distance(k,i)/(1.0 + beta*r_distance(k,i));
        }
    }
}


// funtion to compute derivative in quantum force and Jastrow energy
void VMCSolver::computeJastrowDerivative(int k){
    for(int i=0; i<k; i++){
        double divisor = 1.0 + beta*r_distance(i, k);
        JastrowDerivative(i, k) = a_matrix(i, k)/(divisor*divisor);
    }
    for(int i=k+1; i<nParticles; i++){
        double divisor = 1.0 + beta*r_distance(k, i);
        JastrowDerivative(k, i) = a_matrix(k, i)/(divisor*divisor);
    }
}


// compute laplacian used in Jastrow energy
void VMCSolver::computeJastrowLaplacian(int k){
    for(int i=0; i<k; i++){
        double divisor = 1.0 + beta*r_distance(i, k);
        JastrowLaplacianNew(i, k) = -2.0*a_matrix(i, k)*beta/(divisor*divisor*divisor);
    }
    for(int i=k+1; i<nParticles; i++){
        double divisor = 1.0 + beta*r_distance(k, i);
        JastrowLaplacianNew(k, i) = -2.0*a_matrix(k, i)*beta/(divisor*divisor*divisor);
    }
}


// OBS! Be sure Quantumforce is called before this function is used!!
// compute quantum force
double VMCSolver::computeJastrowEnergy(){
    double sum = 0.0;
    energytermSlaterJastrow = 0.0;

    sum += GradientSquared;
    for(int k=0; k<nParticles; k++){
        for(int i=0; i<k; i++) {
            sum += (nDimensions - 1)/r_distance(i, k)*JastrowDerivative(i, k) + JastrowLaplacianNew(i, k);
        }
        for(int i=k+1; i<nParticles; i++) {
            sum += (nDimensions - 1)/r_distance(k, i)*JastrowDerivative(k, i) + JastrowLaplacianNew(k, i);
        }
    }
    for(int k=0; k<nDimensions; k++){
        energytermSlaterJastrow += dot(SlaterGradientNew.col(k), JastrowGradient.col(k));
    }
    return sum;
}


// efficient algorithm to update slater determinants
void VMCSolver::update_D(mat& D_new, const mat& D_old, int i, int selector){
    i = i - nParticles/2*selector;

    for(int k=0; k<nParticles/2; k++){
        for(int j=0; j<nParticles/2; j++){
            if(j!=i){
                double sum = 0;
                for(int l=0; l<nParticles/2; l++){
                    sum += SlaterPsi(rNew, i + nParticles/2*selector, l)*D_old(l,j);
                }
                D_new(k,j) = D_old(k,j) - D_old(k, i)*sum/R_sd;
            }
            else{
                D_new(k,j) = D_old(k, i)/R_sd;
            }
        }
    }
}


// efficient algorithm to update correlation matrix
void VMCSolver::update_C(mat &CorrelationsMatrix, int k){
    for(int i=0; i<k; i++){
        CorrelationsMatrix(i, k) = a_matrix(i,k)*r_distance(i,k)/(1.0 + beta*r_distance(i,k));
    }
    for(int i=k+1; i<nParticles; i++){
        CorrelationsMatrix(k, i) = a_matrix(k,i)*r_distance(k,i)/(1.0 + beta*r_distance(k,i));
    }
}


// compute the quantum force used in importance sampling
void VMCSolver::QuantumForce(const mat &r, mat &F){
    GradientSquared = 0.0;
    for(int k=0; k<nParticles; k++){
        for(int j=0; j<nDimensions; j++){

            if(activate_JastrowFactor){
                double GradientSum = 0.0;
                for(int i=0; i<k; i++){
                    GradientSum += (r(k,j)-r(i,j))/r_distance(i,k)*JastrowDerivative(i,k);
                }
                for(int i=k+1; i<nParticles; i++){
                    GradientSum -= (r(i,j)-r(k,j))/r_distance(k,i)*JastrowDerivative(k,i);
                }
                F(k,j) =  2.0*(SlaterGradientNew(k,j) + GradientSum);
                GradientSquared += GradientSum*GradientSum;
                JastrowGradient(k, j) = GradientSum;
            }
            else{
                F(k,j) =  2.0*SlaterGradientNew(k,j);
            }
        }
    }
}


double VMCSolver::SlaterPsi(const mat &positions, int i, int j){

    double r;
    double x, y, z;

    r = r_radius(i);

    if(j == 0){
        // 1s hydrogenic orbital
        return exp(-alpha*r);
    }
    else if(j == 1){
        // 2s hydrogenic orbital
        double arg = alpha*r*0.5;
        return (1.0 - arg)*exp(-arg);
    }
    else if(j == 2){
        // 2px hydrogenic orbital
        x = positions(i, 0);
        return x*exp(-alpha*r*0.5);
    }
    else if(j == 3){
        // 2py hydrogenic orbital
        y = positions(i, 1);
        return y*exp(-alpha*r*0.5);
    }
    else if(j == 4){
        // 2px hydrogenic orbital
        z = positions(i, 2);
        return z*exp(-alpha*r*0.5);
    }
    else{
        return 0;
    }
}


// Gradient of orbitals used in quantum force
double VMCSolver::Psi_first_derivative(int i, int j, int k){
    double r, coor;
    double x, y, z;

    r = r_radius(i);
    coor = rNew(i, k);

    x = rNew(i, 0);
    y = rNew(i, 1);
    z = rNew(i, 2);

    if(j == 0){
        return -alpha*coor*exp(-alpha*r)/r;
    }
    else if(j == 1){
        return 0.25*alpha*coor*(alpha*r - 4.0)*exp(-alpha*r*0.5)/r;
    }

    else if(j == 2){
        if(k==0){
            return -(0.5*alpha*x*x - r)*exp(-alpha*r*0.5)/r;
        }
        else if(k==1){
            return -0.5*alpha*x*y*exp(-alpha*r*0.5)/r;
        }
        else{
            return -0.5*alpha*x*z*exp(-alpha*r*0.5)/r;
        }
    }

    else if(j == 3){
        if(k==0){
            return -0.5*alpha*x*y*exp(-alpha*r*0.5)/r;
        }
        else if(k==1){
            return -(0.5*alpha*y*y - r)*exp(-alpha*r*0.5)/r;
        }
        else{
            return -0.5*alpha*y*z*exp(-alpha*r*0.5)/r;
        }
    }

    else if(j == 4){
        if(k==0){
            return -0.5*alpha*x*z*exp(-alpha*r*0.5)/r;
        }
        else if(k==1){
            return -0.5*alpha*y*z*exp(-alpha*r*0.5)/r;
        }
        else{
            return -(0.5*alpha*z*z - r)*exp(-alpha*r*0.5)/r;
        }
    }
    else{
        return 0;
    }
}


// Laplacian of orbitals used in kinetic energy
double VMCSolver::Psi_second_derivative(int i, int j){
    double r;
    double x, y, z;

    r = r_radius(i);
    x = rNew(i, 0);
    y = rNew(i, 1);
    z = rNew(i, 2);

    double r2 = r*r;
    double alpha2 = alpha*alpha;

    if(j == 0){
        return alpha*(alpha*r - 2)*exp(-alpha*r)/r;
    }
    else if(j == 1){
        return -0.125*alpha*(alpha2*r2 - 10.0*alpha*r + 16.0)*exp(-alpha*r*0.5)/r;
    }
    else if(j == 2){
        return 0.25*alpha*x*(alpha*r - 8.0)*exp(-alpha*r*0.5)/r;
    }
    else if(j == 3){
        return 0.25*alpha*y*(alpha*r - 8.0)*exp(-alpha*r*0.5)/r;
    }
    else if(j == 4){
        return 0.25*alpha*z*(alpha*r - 8.0)*exp(-alpha*r*0.5)/r;
    }
    else{
        return 0;
    }
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
    psi2px = alpha*x*exp(-alpha*radius/2.0);
    return psi2px;
}

// 2py hydrogenic orbital
double VMCSolver::psi2py(double &y, double &radius){
    double psi2py;
    psi2py = alpha*y*exp(-alpha*radius/2.0);
    return psi2py;
}

// 2pz hydrogenic orbital
double VMCSolver::psi2pz(double &z, double &radius){
    double psi2pz;
    psi2pz = alpha*z*exp(-alpha*radius/2.0);
    return psi2pz;
}

// compute the Slater determinant for Beryllium
double VMCSolver::SlaterBeryllium(){
    vec argument = zeros(nParticles);
    argument = r_radius;
    // Slater determinant, no factors as they vanish in Metropolis ratio
    double wf  = (psi1s(argument(0))*psi2s(argument(1))-psi1s(argument(1))*psi2s(argument(0)))*(psi1s(argument(2))*psi2s(argument(3))-psi1s(argument(3))*psi2s(argument(2)));
    return wf;
}

// compute the Jastrow factor
double VMCSolver::JastrowMultiplicator(){
    double Psi = 1.0;
    for(int j=0; j < nParticles; j++){
        for(int i=0; i < j; i++){
            Psi *= exp((a_matrix(i,j)*r_distance(i,j))/(1.0 + beta*r_distance(i,j)));
        }
    }
    return Psi;
}

double VMCSolver::waveFunction(const mat &r)
{
    if (AtomType == "helium"){
        if (activate_JastrowFactor){
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

        else{
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
    }

    else if (AtomType == "beryllium"){
        if (activate_JastrowFactor){
            r_func(r);
            double factor = JastrowMultiplicator();
            double hydrogenic = SlaterBeryllium();
            return hydrogenic*factor;
        }
        else{
            r_func(r);
            double hydrogenic = SlaterBeryllium();
            return hydrogenic;
        }
    }
    else{
        return(0);
    }
}




/*
double VMCSolver::fillSolidHarmonics(int ml, int l){
    mat SolidHarmonics = zeros(7, 4);

    double r;
    double x, y, z;

    r = r_radius(i);
    x = rNew(i, 0);
    y = rNew(i, 1);
    z = rNew(i, 2);

    if(ml == 3){
        if(l == 0){
            return 0;
        }
        else if(l == 1){
            return 0;
        }
        else if(l == 2){
            return 0;
        }
        else{
            return 0.5*sqrt(5.0/2.0)*(x*x - 3*y*y)*x;
        }
    }

    else if(ml == 2){
        if(l == 0){
            return 0;
        }
        else if(l == 1){
            return 0;
        }
        else if(l == 2){
            return 0.5*sqrt(3.0)*(x*x - y*y);
        }
        else{
            return 0.5*sqrt(15.0)*(x*x - y*y)*z;
        }
    }

    else if(ml == 1){
        if(l == 0){
            return 0;
        }
        else if(l == 1){
            return x;
        }
        else if(l == 2){
            return sqrt(3.0)*x*z;
        }
        else{
            return 0.5*sqrt(3.0/2.0)*(5.0*z*z - r*r)*x;
        }
    }

    else if(ml == 0){
        if(l == 0){
            return 0;
        }
        else if(l == 1){
            return y;
        }
        else if(l == 2){
            return 0.5*(3.0*z*z - r*r);
        }
        else{
            return 0.5*(5*z*z - 3.0*r*r)*x;
        }
    }

    else if(ml == -1){
        if(l == 0){
            return 0;
        }
        else if(l == 1){
            return z;
        }
        else if(l == 2){
            return sqrt(3.0)*y*z;
        }
        else{
            return 0.5*sqrt(3.0/2.0)*(5.0*z*z - r*r)*y;
        }
    }

    else if(ml == -2){
        if(l == 0){
            return 0;
        }
        else if(l == 1){
            return 0;
        }
        else if(l == 2){
            return sqrt(3.0)*x*y;
        }
        else{
            return sqrt(15.0)*x*y*z;
        }
    }

    else{
        if(l == 0){
            return 0;
        }
        else if(l == 1){
            return 0;
        }
        else if(l == 2){
            return 0;
        }
        else{
            return 0.5*sqrt(5.0/2.0)*(3*x*x - y*y)*y;
        }
    }

}

*/
