#ifndef GAUSSIAN_H
#define GAUSSIAN_H

void ReadFile_fillGTO(mat &GTO_mat, string filename);
void fillGTO();

void SlaterDeterminantGaussian();
void compute_R_sd_gaussian(int i);
void update_D_gaussian(mat& D_new, const mat& D_old, int i, int selector);
void SlaterGradientGaussian(int i);
double SlaterLaplacianGaussian();
double SlaterPsiGaussian(int particle, int orb_select);
double PsiGaussian_derivative(int particle, int orb_select, int dimension);
double PsiGaussian_laplacian(int particle, int orb_select);

double factorial_func(int number);
double Normalization_factor(double GTO_alpha, int i, int j, int k);
double G_func(double GTO_alpha, int particle, int i, int j, int k);
double G_derivative(double GTO_alpha, int particle, int orb_select, int dimension, int i, int j, int k);
double G_laplacian(double GTO_alpha, int particle, int orb_select, int i, int j, int k);

double GaussianOrbitals(int i, int j);


#endif // GAUSSIAN_H
