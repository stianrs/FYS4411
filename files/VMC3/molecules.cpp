// This file contain all Slater functions and orbitals needed for using hydrogenic wave functions on molecules

#include "molecules.h"
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

// compute the Slater determinant for the first time
void VMCSolver::SlaterDeterminant(){
    mat D_up = zeros(nParticles/2, nParticles/2);
    mat D_down = zeros(nParticles/2, nParticles/2);

    // compute spinn up part and spin down part
    for(int j=0; j<nParticles/2; j++){
        for(int i=0; i<nParticles/2; i++){
            D_up(i,j) = SlaterPsi(i, j);
            D_down(i,j) = SlaterPsi(i+(nParticles/2), j);
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
            R_sd += SlaterPsi(i, j)*D_up_old(j, i);
        }
    }
    else{
        for(int j=0; j<nParticles/2; j++){
            R_sd += SlaterPsi(i, j)*D_down_old(j, i-nParticles/2);
        }
    }
}


// efficient algorithm to update slater determinants
void VMCSolver::update_D(mat& D_new, const mat& D_old, int i, int selector){
    i = i - nParticles/2*selector;

    for(int k=0; k<nParticles/2; k++){
        for(int j=0; j<nParticles/2; j++){
            if(j!=i){
                double sum = 0;
                for(int l=0; l<nParticles/2; l++){
                    sum += SlaterPsi(i + nParticles/2*selector, l)*D_old(l,j);
                }
                D_new(k,j) = D_old(k,j) - D_old(k, i)*sum/R_sd;
            }
            else{
                D_new(k,j) = D_old(k, i)/R_sd;
            }
        }
    }
}


// compute slater first derivative
void VMCSolver::SlaterGradient(int i){
    if(i < nParticles/2){
        for(int k=0; k<nDimensions; k++){
            double derivative_up = 0.0;
            for(int j=0; j<nParticles/2; j++){
                derivative_up += Psi_derivative(i, j, k)*D_up_old(j, i);
            }
            SlaterGradientNew(i, k) = (1.0/R_sd)*derivative_up;
        }
    }
    else{
        for(int k=0; k<nDimensions; k++){
            double derivative_down = 0.0;
            for(int j=0; j<nParticles/2; j++){
                derivative_down += Psi_derivative(i, j, k)*D_down_old(j, i-nParticles/2);
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
            derivative_up += Psi_laplacian(i, j)*D_up_new(j, i);
            derivative_down += Psi_laplacian(i+nParticles/2, j)*D_down_new(j, i);
        }
    }
    double derivative_sum = derivative_up + derivative_down;
    return derivative_sum;
}


// Hydrogenic orbitals for molecules
double VMCSolver::SlaterPsi(int i, int j){

    double r1, r2;
    r1 = r_radius(i);
    r2 = r_radius2(i);

    if(j == 0){
        return exp(-alpha*r1) + exp(-alpha*r2);
    }
    else if(j == 1){
        return exp(-alpha*r1) - exp(-alpha*r2);
    }
    else if(j == 2){
        double arg1 = alpha*r1*0.5;
        double arg2 = alpha*r2*0.5;
        return (1.0 - arg1)*exp(-arg1) + (1.0 - arg2)*exp(-arg2);
    }
    else if(j == 3){
        double arg1 = alpha*r1*0.5;
        double arg2 = alpha*r2*0.5;
        return (1.0 - arg1)*exp(-arg1) - (1.0 - arg2)*exp(-arg2);
    }
    else{
        cout << "Reached end of orbitals!" << endl;
        exit(1);
    }
}


// Gradient of orbitals used in quantum force
double VMCSolver::Psi_derivative(int i, int j, int k){
    double r1, r2, R_2, coor;

    r1 = r_radius(i);
    r2 = r_radius2(i);
    R_2 = R_molecule/2.0;
    coor = rNew(i, k);

    if(j == 0){
        if(k < 2){
            return (-alpha*coor*exp(-alpha*r1)/r1) + (-alpha*coor*exp(-alpha*r2)/r2);
        }
        else{
            return (-alpha*(coor + R_2)*exp(-alpha*r1)/r1) + (-alpha*(coor - R_2)*exp(-alpha*r2)/r2);
        }
    }
    if(j == 1){
        if(k < 2){
            return (-alpha*coor*exp(-alpha*r1)/r1) - (-alpha*coor*exp(-alpha*r2)/r2);
        }
        else{
            return (-alpha*(coor + R_2)*exp(-alpha*r1)/r1) - (-alpha*(coor - R_2)*exp(-alpha*r2)/r2);
        }
    }
    else if(j == 2){
        if(k < 2){
            return (0.25*alpha*coor*(alpha*r1 - 4.0)*exp(-alpha*r1*0.5)/r1) + (0.25*alpha*coor*(alpha*r2 - 4.0)*exp(-alpha*r2*0.5)/r2);
        }
        else{
            return (0.25*alpha*(coor + R_2)*(alpha*r1 - 4.0)*exp(-alpha*r1*0.5)/r1) + (0.25*alpha*(coor - R_2)*(alpha*r2 - 4.0)*exp(-alpha*r2*0.5)/r2);
        }
    }
    else if(j == 3){
        if(k < 2){
            return (0.25*alpha*coor*(alpha*r1 - 4.0)*exp(-alpha*r1*0.5)/r1) - (0.25*alpha*coor*(alpha*r2 - 4.0)*exp(-alpha*r2*0.5)/r2);
        }
        else{
            return (0.25*alpha*(coor + R_2)*(alpha*r1 - 4.0)*exp(-alpha*r1*0.5)/r1) - (0.25*alpha*(coor - R_2)*(alpha*r2 - 4.0)*exp(-alpha*r2*0.5)/r2);
        }
    }
    else{
        cout << "Reached end of orbitals!" << endl;
        exit(1);
    }
}


// Laplacian of orbitals used in kinetic energy
double VMCSolver::Psi_laplacian(int i, int j){
    double r1, r2, alpha2;

    r1 = r_radius(i);
    r2 = r_radius2(i);
    alpha2 = alpha*alpha;

    if(j == 0){
        return (alpha*(alpha*r1 - 2.0)*exp(-alpha*r1)/r1) + (alpha*(alpha*r2 - 2.0)*exp(-alpha*r2)/r2);
    }
    else if(j == 1){
        return (alpha*(alpha*r1 - 2.0)*exp(-alpha*r1)/r1) - (alpha*(alpha*r2 - 2.0)*exp(-alpha*r2)/r2);
    }
    else if(j == 2){
        return (-0.125*alpha*(alpha2*r1*r1 - 10.0*alpha*r1 + 16.0)*exp(-alpha*r1*0.5)/r1) + (-0.125*alpha*(alpha2*r2*r2 - 10.0*alpha*r2 + 16.0)*exp(-alpha*r2*0.5)/r2);
    }
    else if(j == 3){
        return (-0.125*alpha*(alpha2*r1*r1 - 10.0*alpha*r1 + 16.0)*exp(-alpha*r1*0.5)/r1) - (-0.125*alpha*(alpha2*r2*r2 - 10.0*alpha*r2 + 16.0)*exp(-alpha*r2*0.5)/r2);
    }
    else{
        cout << "Reached end of orbitals!" << endl;
        exit(1);
    }
}


