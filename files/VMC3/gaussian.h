#ifndef GAUSSIAN_H
#define GAUSSIAN_H


#include <armadillo>
#include <fstream>
#include <iostream>

using namespace std;
using namespace arma;

void SlaterDeterminant();
void compute_R_sd(int i);
void update_D(mat &D_new, const mat &D_old, int i, int selector);
void SlaterGradient(int i);
double SlaterLaplacian();
double SlaterPsi(int particle, int orb_select);
double Psi_derivative(int particle, int orb_select, int dimension);
double Psi_laplacian(int particle, int orb_select);

double factorial_func(int number);
double Normalization_factor(double GTO_alpha, int i, int j, int k);
double G_func(double GTO_alpha, int particle, int i, int j, int k);
double G_derivative(double GTO_alpha, int particle, int orb_select, int dimension, int i, int j, int k);
double G_laplacian(double GTO_alpha, int particle, int orb_select, int i, int j, int k);

double GaussianOrbitals(int i, int j);


#endif // GAUSSIAN_H
