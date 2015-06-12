
#include "gaussian.h"

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
const double pi = 4*atan(1.0);


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


// Compute the R_sd ratio
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


// Efficient algorithm to update slater determinants
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


// Compute slater first derivative
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


// Compute slater second derivative
double VMCSolver::SlaterLaplacian(){
    double derivative_up = 0.0;
    double derivative_down = 0.0;

    for(int i=0; i<nParticles/2; i++){
        for(int j=0; j<nParticles/2; j++){
            derivative_up += Psi_laplacian  (i, j)*D_up_new(j, i);
            derivative_down += Psi_laplacian(i+nParticles/2, j)*D_down_new(j, i);
        }
    }
    double derivative_sum = derivative_up + derivative_down;
    return derivative_sum;
}


// Called for setting up Slater determinant with GTO
double VMCSolver::SlaterPsi(int particle, int orb){
    double psi = 0.0;

    if(AtomType=="He"){
        double alpha1 = GTO_values(0, 0);
        double alpha2 = GTO_values(1, 0);
        double alpha3 = GTO_values(2, 0);

        double c1 = GTO_values(0, 1);
        double c2 = GTO_values(1, 1);
        double c3 = GTO_values(2, 1);

        double K1 = GTO_coef_values(0, 0);
        double K2 = GTO_coef_values(1, 0);

        psi = K1*(c1*G_func(alpha1, particle, 0, 0, 0)) + K2*(c2*G_func(alpha2, particle, 0, 0, 0) + c3*G_func(alpha3, particle, 0, 0, 0));
    }
    else{
        int Kp_set;
        Kp_set = orb;

        double alpha1 = GTO_values(0, 0);
        double alpha2 = GTO_values(1, 0);
        double alpha3 = GTO_values(2, 0);
        double alpha4 = GTO_values(3, 0);
        double alpha5 = GTO_values(4, 0);
        double alpha6 = GTO_values(5, 0);
        double alpha7 = GTO_values(6, 0);
        double alpha8 = GTO_values(7, 0);
        double alpha9 = GTO_values(8, 0);

        double c1 = GTO_values(0, 1);
        double c2 = GTO_values(1, 1);
        double c3 = GTO_values(2, 1);
        double c4 = GTO_values(3, 1);
        double c5 = GTO_values(4, 1);
        double c6 = GTO_values(5, 1);
        double c7 = GTO_values(6, 1);
        double c8 = GTO_values(7, 1);
        double c9 = GTO_values(8, 1);

        double K1 = GTO_coef_values(0, Kp_set);
        double K2 = GTO_coef_values(1, Kp_set);
        double K3 = GTO_coef_values(2, Kp_set);
        double K4 = GTO_coef_values(3, Kp_set);
        double K5 = GTO_coef_values(4, Kp_set);
        double K6 = GTO_coef_values(5, Kp_set);
        double K7 = GTO_coef_values(6, Kp_set);
        double K8 = GTO_coef_values(7, Kp_set);
        double K9 = GTO_coef_values(8, Kp_set);

        psi = K1*(c1*G_func(alpha1, particle, 0, 0, 0) + c2*G_func(alpha2, particle, 0, 0, 0) + c3*G_func(alpha3, particle, 0, 0, 0))
                + K2*(c4*G_func(alpha4, particle, 0, 0, 0) + c5*G_func(alpha5, particle, 0, 0, 0)) + K3*(c6*G_func(alpha6, particle, 0, 0, 0))
                + K4*(c7*G_func(alpha7, particle, 1, 0, 0) + c8*G_func(alpha8, particle, 1, 0, 0)) + K5*(c9*G_func(alpha9, particle, 1, 0, 0))
                + K6*(c7*G_func(alpha7, particle, 0, 1, 0) + c8*G_func(alpha8, particle, 0, 1, 0)) + K7*(c9*G_func(alpha9, particle, 0, 1, 0))
                + K8*(c7*G_func(alpha7, particle, 0, 0, 1) + c8*G_func(alpha8, particle, 0, 0, 1)) + K9*(c9*G_func(alpha9, particle, 0, 0, 1));
    }
    return psi;
}


// Called for setting up slatergradient
double VMCSolver::Psi_derivative(int particle, int orb, int dimension){
    double psi = 0.0;

    if(AtomType=="He"){
        double alpha1 = GTO_values(0, 0);
        double alpha2 = GTO_values(1, 0);
        double alpha3 = GTO_values(2, 0);

        double c1 = GTO_values(0, 1);
        double c2 = GTO_values(1, 1);
        double c3 = GTO_values(2, 1);

        double K1 = GTO_coef_values(0, 0);
        double K2 = GTO_coef_values(1, 0);

        psi = K1*(c1*G_der(alpha1, particle, orb, dimension, 0, 0, 0))
                + K2*(c2*G_der(alpha2, particle, orb, dimension, 0, 0, 0) + c3*G_der(alpha3, particle, orb, dimension, 0, 0, 0));
    }
    else{
        int Kp_set;
        Kp_set = orb;

        double alpha1 = GTO_values(0, 0);
        double alpha2 = GTO_values(1, 0);
        double alpha3 = GTO_values(2, 0);
        double alpha4 = GTO_values(3, 0);
        double alpha5 = GTO_values(4, 0);
        double alpha6 = GTO_values(5, 0);
        double alpha7 = GTO_values(6, 0);
        double alpha8 = GTO_values(7, 0);
        double alpha9 = GTO_values(8, 0);

        double c1 = GTO_values(0, 1);
        double c2 = GTO_values(1, 1);
        double c3 = GTO_values(2, 1);
        double c4 = GTO_values(3, 1);
        double c5 = GTO_values(4, 1);
        double c6 = GTO_values(5, 1);
        double c7 = GTO_values(6, 1);
        double c8 = GTO_values(7, 1);
        double c9 = GTO_values(8, 1);

        double K1 = GTO_coef_values(0, Kp_set);
        double K2 = GTO_coef_values(1, Kp_set);
        double K3 = GTO_coef_values(2, Kp_set);
        double K4 = GTO_coef_values(3, Kp_set);
        double K5 = GTO_coef_values(4, Kp_set);
        double K6 = GTO_coef_values(5, Kp_set);
        double K7 = GTO_coef_values(6, Kp_set);
        double K8 = GTO_coef_values(7, Kp_set);
        double K9 = GTO_coef_values(8, Kp_set);

        psi = K1*(c1*G_der(alpha1, particle, orb, dimension, 0, 0, 0) + c2*G_der(alpha2, particle, orb, dimension, 0, 0, 0) + c3*G_der(alpha3, particle, orb, dimension, 0, 0, 0))
                + K2*(c4*G_der(alpha4, particle, orb, dimension, 0, 0, 0) + c5*G_der(alpha5, particle, orb, dimension, 0, 0, 0)) + K3*(c6*G_der(alpha6, particle, orb, dimension, 0, 0, 0))
                + K4*(c7*G_der(alpha7, particle, orb, dimension, 1, 0, 0) + c8*G_der(alpha8, particle, orb, dimension, 1, 0, 0)) + K5*(c9*G_der(alpha9, particle, orb, dimension, 1, 0, 0))
                + K6*(c7*G_der(alpha7, particle, orb, dimension, 0, 1, 0) + c8*G_der(alpha8, particle, orb, dimension, 0, 1, 0)) + K7*(c9*G_der(alpha9, particle, orb, dimension, 0, 1, 0))
                + K8*(c7*G_der(alpha7, particle, orb, dimension, 0, 0, 1) + c8*G_der(alpha8, particle, orb, dimension, 0, 0, 1)) + K9*(c9*G_der(alpha9, particle, orb, dimension, 0, 0, 1));
    }
    return psi;
}


// Called for setting up slaterlaplacian
double VMCSolver::Psi_laplacian(int particle, int orb){
    double psi = 0.0;

    if(AtomType=="He"){
        double alpha1 = GTO_values(0, 0);
        double alpha2 = GTO_values(1, 0);
        double alpha3 = GTO_values(2, 0);

        double c1 = GTO_values(0, 1);
        double c2 = GTO_values(1, 1);
        double c3 = GTO_values(2, 1);

        double K1 = GTO_coef_values(0, 0);
        double K2 = GTO_coef_values(1, 0);

        psi = K1*(c1*G_lap(alpha1, particle, orb, 0, 0, 0)) + K2*(c2*G_lap(alpha2, particle, orb, 0, 0, 0) + c3*G_lap(alpha3, particle, orb, 0, 0, 0));
    }
    else{
        int Kp_set;
        Kp_set = orb;

        double alpha1 = GTO_values(0, 0);
        double alpha2 = GTO_values(1, 0);
        double alpha3 = GTO_values(2, 0);
        double alpha4 = GTO_values(3, 0);
        double alpha5 = GTO_values(4, 0);
        double alpha6 = GTO_values(5, 0);
        double alpha7 = GTO_values(6, 0);
        double alpha8 = GTO_values(7, 0);
        double alpha9 = GTO_values(8, 0);

        double c1 = GTO_values(0, 1);
        double c2 = GTO_values(1, 1);
        double c3 = GTO_values(2, 1);
        double c4 = GTO_values(3, 1);
        double c5 = GTO_values(4, 1);
        double c6 = GTO_values(5, 1);
        double c7 = GTO_values(6, 1);
        double c8 = GTO_values(7, 1);
        double c9 = GTO_values(8, 1);

        double K1 = GTO_coef_values(0, Kp_set);
        double K2 = GTO_coef_values(1, Kp_set);
        double K3 = GTO_coef_values(2, Kp_set);
        double K4 = GTO_coef_values(3, Kp_set);
        double K5 = GTO_coef_values(4, Kp_set);
        double K6 = GTO_coef_values(5, Kp_set);
        double K7 = GTO_coef_values(6, Kp_set);
        double K8 = GTO_coef_values(7, Kp_set);
        double K9 = GTO_coef_values(8, Kp_set);

        psi = K1*(c1*G_lap(alpha1, particle, orb, 0, 0, 0) + c2*G_lap(alpha2, particle, orb, 0, 0, 0) + c3*G_lap(alpha3, particle, orb, 0, 0, 0))
                + K2*(c4*G_lap(alpha4, particle, orb, 0, 0, 0) + c5*G_lap(alpha5, particle, orb, 0, 0, 0)) + K3*(c6*G_lap(alpha6, particle, orb, 0, 0, 0))
                + K4*(c7*G_lap(alpha7, particle, orb, 1, 0, 0) + c8*G_lap(alpha8, particle, orb, 1, 0, 0)) + K5*(c9*G_lap(alpha9, particle, orb, 1, 0, 0))
                + K6*(c7*G_lap(alpha7, particle, orb, 0, 1, 0) + c8*G_lap(alpha8, particle, orb, 0, 1, 0)) + K7*(c9*G_lap(alpha9, particle, orb, 0, 1, 0))
                + K8*(c7*G_lap(alpha7, particle, orb, 0, 0, 1) + c8*G_lap(alpha8, particle, orb, 0, 0, 1)) + K9*(c9*G_lap(alpha9, particle, orb, 0, 0, 1));
    }
    return psi;
}


// Computes factorial of a given number
double VMCSolver::factorial_func(int number){
    int factorial = 1;
    for(int i=2; i<=number; i++){
        factorial *= i;
    }
    return factorial;
}


// General expression for computing the normalization factor in GTO orbitals
double VMCSolver::Normalization_factor(double GTO_alpha, int i, int j, int k){
    int fac_i = factorial_func(i);
    int fac_j = factorial_func(j);
    int fac_k = factorial_func(k);
    int fac_2i = factorial_func(2*i);
    int fac_2j = factorial_func(2*j);
    int fac_2k = factorial_func(2*k);

    double frac_over = pow(8.0*GTO_alpha, i+j+k) * (fac_i*fac_j*fac_k);
    double frac_under = fac_2i*fac_2j*fac_2k;

    double N = pow(((2.0*GTO_alpha)/pi), (3.0/4.0)) * sqrt(frac_over/frac_under);
    return N;
}


// Compute G in GTO orbitals for given parameters
double VMCSolver::G_func(double GTO_alpha, int particle, int i, int j, int k){
    double x, y, z, r, N;

    x = rNew(particle, 0);
    y = rNew(particle, 1);
    z = rNew(particle, 2);

    N = Normalization_factor(GTO_alpha, i, j, k);

    double G = N*pow(x, i)*pow(y, j)*pow(z, k)*exp(-GTO_alpha*(x*x + y*y + z*z));
    return G;
}


// Compute G derivative in GTO orbitals for given parameters
double VMCSolver::G_der(double GTO_alpha, int particle, int orb_select, int dimension, int i, int j, int k){
    double x, y, z, r, N, coor, gaussian;

    coor = rNew(particle, dimension);
    x = rNew(particle, 0);
    y = rNew(particle, 1);
    z = rNew(particle, 2);
    r = r_radius(particle);

    gaussian = exp(-GTO_alpha*r*r);
    N = Normalization_factor(GTO_alpha, i, j, k);

    if(orb_select < 2){
        double Factor = -2.0*N*GTO_alpha;
        double G = Factor*coor*gaussian;
        return G;
    }
    else if(orb_select == 2){
        if(dimension == 0){
            double G = -N*(2.0*GTO_alpha*x*x - 1.0)*gaussian;
            return G;
        }
        else if(dimension == 1){
            double G = -2.0*N*GTO_alpha*x*y*gaussian;
            return G;
        }
        else if(dimension == 2){
            double G = -2.0*N*GTO_alpha*x*z*gaussian;
            return G;
        }
    }
    else if(orb_select == 3){
        if(dimension == 0){
            double G = -2.0*N*GTO_alpha*x*y*gaussian;
            return G;
        }
        else if(dimension == 1){
            double G = -N*(2.0*GTO_alpha*y*y - 1.0)*gaussian;
            return G;
        }
        else if(dimension == 2){
            double G = -2.0*N*GTO_alpha*y*z*gaussian;
            return G;
        }
    }
    else if(orb_select == 4){
        if(dimension == 0){
            double G = -2.0*N*GTO_alpha*x*z*gaussian;
            return G;
        }
        else if(dimension == 1){
            double G = -2.0*N*GTO_alpha*y*z*gaussian;
            return G;
        }
        else if(dimension == 2){
            double G = -N*(2.0*GTO_alpha*z*z - 1.0)*gaussian;
            return G;
        }
    }
    else{
        cout << "Reached end of orbitals!" << endl;
        exit(1);
    }
}


// Compute G laplacian in GTO orbitals for given parameters
double VMCSolver::G_lap(double GTO_alpha, int particle, int orb_select, int i, int j, int k){
    double x, y, z;
    double r;
    double N;

    x = rNew(particle, 0);
    y = rNew(particle, 1);
    z = rNew(particle, 2);
    r = r_radius(particle);

    N = Normalization_factor(GTO_alpha, i, j, k);

    if(orb_select < 2){
        double G = 2.0*N*GTO_alpha*(2.0*GTO_alpha*r*r - 3.0)*exp(-GTO_alpha*r*r);
        return G;
    }
    else if(orb_select == 2){
        double G = 2.0*N*GTO_alpha*x*(2.0*GTO_alpha*r*r - 5.0)*exp(-GTO_alpha*r*r);
        return G;
    }
    else if(orb_select == 3){
        double G = 2.0*N*GTO_alpha*y*(2.0*GTO_alpha*r*r - 5.0)*exp(-GTO_alpha*r*r);
        return G;
    }
    else if(orb_select == 4){
        double G = 2.0*N*GTO_alpha*z*(2.0*GTO_alpha*r*r - 5.0)*exp(-GTO_alpha*r*r);
        return G;
    }
    else{
        cout << "Reached end of orbitals!" << endl;
        exit(1);
    }
}


// Function to compute GTO (gaussian orbitals)
double VMCSolver::GaussianOrbitals(int i, int j){
    double psi_val = SlaterPsi(i, j);
    return psi_val;
}


