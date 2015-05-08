#include "gaussian.h"

#include "vmcsolver.h"
#include "lib.h"
#include "investigate.h"
#include "hydrogenic.h"

#include <armadillo>
#include <iostream>
#include <time.h>
#include <fstream>
#include <mpi.h>

using namespace arma;
using namespace std;


// Reads a file and store values in a matrix
void VMCSolver::ReadFile_fillGTO(mat &GTO_mat, string filename){
    int num_rows = GTO_mat.n_rows;
    double GTO_alpha, GTO_c1, GTO_c2;
    mat GTO_values = zeros(num_rows, 3);
    ifstream myfile;

    myfile.open(filename.c_str(), ios::in);
    if(!myfile){
        cout << "Not able to open file!" << endl;
        exit(1);
    }
    int k = 0;
    while(k<num_rows){
        myfile >> GTO_alpha >> GTO_c1 >> GTO_c2;
        GTO_values(k, 0) = GTO_alpha;
        GTO_values(k, 1) = GTO_c1;
        GTO_values(k, 2) = GTO_c2;
        k++;
    }
    myfile.close();
    GTO_mat = GTO_values;
}


// Fill GTO_matrices for Helium, Beryllium and Neon with 3-21G basis set
void VMCSolver::fillGTO(){
    int num_rows_helium = 3; int num_rows_beryllium = 6; int num_rows_neon = 6;
    GTO_helium = zeros<mat>(num_rows_helium, 3);
    GTO_beryllium = zeros<mat>(num_rows_beryllium, 3);
    GTO_neon = zeros<mat>(num_rows_neon, 3);
    ReadFile_fillGTO(GTO_helium, "GTO_helium.dat");
    ReadFile_fillGTO(GTO_beryllium, "GTO_beryllium.dat");
    ReadFile_fillGTO(GTO_neon, "GTO_neon.dat");

    if(AtomType == "helium"){
        int num_rows = GTO_helium.n_rows;
        GTO_values = zeros(num_rows, 3);
        GTO_values = GTO_helium;
    }
    else if(AtomType == "beryllium"){
        int num_rows = GTO_beryllium.n_rows;
        GTO_values = zeros(num_rows, 3);
        GTO_values = GTO_beryllium;
    }
    else if(AtomType == "neon"){
        int num_rows = GTO_neon.n_rows;
        GTO_values = zeros(num_rows, 3);
        GTO_values = GTO_neon;
    }
}


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
            derivative_up += Psi_laplacian(i, j)*D_up_new(j, i);
            derivative_down += Psi_laplacian(i+nParticles/2, j)*D_down_new(j, i);
        }
    }
    double derivative_sum = derivative_up + derivative_down;
    return derivative_sum;
}


// Called for setting up slaterdeterminant
double VMCSolver::SlaterPsi(int particle, int orb_select){
    int sum_num;
    int offset1;
    int offset2;
    double GTO_alpha;
    double Kp = 0.5;
    double phi = 0.0;
    double psi;
    int i = 0; int j = 0; int k = 0;

    if(orb_select >= 2){
        if(orb_select==2){
            i = 1;
        }
        else if(orb_select==3){
            j = 1;
        }
        else if(orb_select >=4 ){
            k = 1;
        }
    }
    if(orb_select==0){
        sum_num = 3;
        for(int index=0; index<sum_num; index++){
            GTO_alpha = GTO_values(index, 0);
            phi += GTO_values(index, 1)*G_func(GTO_alpha, particle, i, j, k);
        }
    }
    else{
        sum_num = 2;
        offset1 = 3;
        offset2 = 5;

        for(int index = offset1; index < sum_num+offset1; index++){
            GTO_alpha = GTO_values(index, 0);
            phi += GTO_values(index, 1+i+j+k)*G_func(GTO_alpha, particle, i, j, k);
        }
        phi = Kp*phi;

        GTO_alpha = GTO_values(offset2, 0);
        phi += Kp*GTO_values(offset2, 1+i+j+k)*G_func(GTO_alpha, particle, i, j, k);
    }
    psi = phi;
    return psi;
}


// Called for setting up slatergradient
double VMCSolver::Psi_derivative(int particle, int orb_select, int dimension){
    int sum_num;
    int offset1;
    int offset2;
    double GTO_alpha;
    double Kp = 0.5;
    double phi = 0.0;
    double psi;
    int i = 0; int j = 0; int k = 0;

    if(orb_select >= 2){
        if(orb_select==2){
            i = 1;
        }
        else if(orb_select==3){
            j = 1;
        }
        else if(orb_select >=4 ){
            k = 1;
        }
    }
    if(orb_select==0){
        sum_num = 3;
        for(int index=0; index<sum_num; index++){
            GTO_alpha = GTO_values(index, 0);
            phi += GTO_values(index, 1)*G_derivative(GTO_alpha, particle, orb_select, dimension, i, j, k);
        }
    }
    else{
        sum_num = 2;
        offset1 = 3;
        offset2 = 5;

        for(int index = offset1; index < sum_num+offset1; index++){
            GTO_alpha = GTO_values(index, 0);
            phi += GTO_values(index, 1+i+j+k)*G_derivative(GTO_alpha, particle, orb_select, dimension, i, j, k);
        }
        phi = Kp*phi;

        GTO_alpha = GTO_values(offset2, 0);
        phi += Kp*GTO_values(offset2, 1+i+j+k)*G_derivative(GTO_alpha, particle, orb_select, dimension, i, j, k);
    }
    psi = phi;
    return psi;
}


// Called for setting up slaterlaplacian
double VMCSolver::Psi_laplacian(int particle, int orb_select){
    int sum_num;
    int offset1;
    int offset2;
    double GTO_alpha;
    double Kp = 0.5;
    double phi = 0.0;
    double psi;
    int i = 0; int j = 0; int k = 0;

    if(orb_select >= 2){
        if(orb_select==2){
            i = 1;
        }
        else if(orb_select==3){
            j = 1;
        }
        else if(orb_select >=4 ){
            k = 1;
        }
    }
    if(orb_select==0){
        sum_num = 3;
        for(int index=0; index<sum_num; index++){
            GTO_alpha = GTO_values(index, 0);
            phi += GTO_values(index, 1)*G_laplacian(GTO_alpha, particle, orb_select, i, j, k);
        }
    }
    else{
        sum_num = 2;
        offset1 = 3;
        offset2 = 5;

        for(int index = offset1; index < sum_num+offset1; index++){
            GTO_alpha = GTO_values(index, 0);
            phi += GTO_values(index, 1+i+j+k)*G_laplacian(GTO_alpha, particle, orb_select, i, j, k);
        }
        phi = Kp*phi;

        GTO_alpha = GTO_values(offset2, 0);
        phi += Kp*GTO_values(offset2, 1+i+j+k)*G_laplacian(GTO_alpha, particle, orb_select, i, j, k);
    }
    psi = phi;
    return psi;
}


// Computes fatorial of a given number
double VMCSolver::factorial_func(int number){
    int factorial = 1;
    for(int i=number; i>0; i--){
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

    double frac_over = pow((8.0*GTO_alpha), (i+j+k))*fac_i*fac_j*fac_k;
    double frac_under = fac_2i*fac_2j*fac_2k;

    double N = pow(((2.0*GTO_alpha)/pi), (3.0/4.0)) * pow((frac_over/frac_under), (1.0/2.0));
    return N;
}


// Compute G in GTO orbitals for given parameters
double VMCSolver::G_func(double GTO_alpha, int particle, int i, int j, int k){
    double x, y, z, r, N;

    x = rNew(particle, 0);
    y = rNew(particle, 1);
    z = rNew(particle, 2);
    r = r_radius(particle);

    N = Normalization_factor(GTO_alpha, i, j, k);

    double G = N*pow(x, i)*pow(y, j)*pow(z, k)*exp(-GTO_alpha*r*r);
    return G;
}

// Compute G derivative in GTO orbitals for given parameters
double VMCSolver::G_derivative(double GTO_alpha, int particle, int orb_select, int dimension, int i, int j, int k){
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
        return 0;
    }
}


// Compute G laplacian in GTO orbitals for given parameters
double VMCSolver::G_laplacian(double GTO_alpha, int particle, int orb_select, int i, int j, int k){
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
        return 0;
    }
}


// Function to compute GTO (gaussian orbitals)
double VMCSolver::GaussianOrbitals(int i, int j){
    double psi_val = SlaterPsi(i, j);
    return psi_val;
}


