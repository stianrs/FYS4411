
#ifndef MOLECULES_H
#define MOLECULES_H

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

#endif // MOLECULES_H

