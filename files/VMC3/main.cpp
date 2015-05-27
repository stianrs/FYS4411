/*
This is the main program used to run Monte Carlo simulations with "vmcsolver.cpp" and run different functions
in "investigate.cpp" using vmcsolver.cpp.
*/

#include "vmcsolver.h"
#include "investigate.h"
#include "hydrogenic.h"
#include "gaussian.h"


#include <iostream>
#include <armadillo>

#include <mpi.h>



using namespace std;
using namespace arma;

int main(int nargs, char *args[])
{
    int nCycles = 1e7;

    int my_rank, world_size;
    double energy, sum_energy;

    // Initialize the parallel environment
    MPI_Init (&nargs, &args);
    MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size (MPI_COMM_WORLD, &world_size);

    VMCSolver *solver = new VMCSolver();
    energy = solver->runMonteCarloIntegration(nCycles, my_rank, world_size);

    // Collect energy estimates
    MPI_Reduce(&energy, &sum_energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Final energy estimate with multiple processors
    if(my_rank==0){
        cout << "Totalt cycles: " << nCycles << endl;
        cout << endl << "Ground state estimate: " << sum_energy/world_size << endl;
    }

    //VMCSolver *investigateCPU = new VMCSolver();
    //investigateCPU->InvestigateCPUtime(my_rank, world_size);

    //VMCSolver *investigateParameters = new VMCSolver();
    //investigateParameters->InvestigateOptimalParameters(my_rank, world_size);

    //VMCSolver *Blocking = new VMCSolver();
    //Blocking->BlockingFunc(my_rank, world_size);

    //VMCSolver *investigateOnebodyDensity_ChargeDensity = new VMCSolver();
    //investigateOnebodyDensity_ChargeDensity ->OnebodyDensity_ChargeDensity(my_rank, world_size);

    //VMCSolver *investigateR_dependence_molecules = new VMCSolver();
    //investigateR_dependence_molecules ->R_dependence_molecules(my_rank, world_size);


    // Clean up and close the MPI environment
    MPI_Finalize();



    /*

    VMCSolver *investigateAlpha = new VMCSolver();
    investigateAlpha->InvestigateOptimalAlpha();

    VMCSolver *investigateParameters = new VMCSolver();
    investigateParameters->InvestigateOptimalParameters();

    VMCSolver *investigateVariance = new VMCSolver();
    investigateVariance->InvestigateVarianceNcycles();

    VMCSolver *Blocking = new VMCSolver();
    Blocking->BlockingFunc();

    VMCSolver *investigateCPU = new VMCSolver();
    investigateCPU->InvestigateCPUtime();

    VMCSolver *investigateTimestepDependence = new VMCSolver();
    investigateTimestepDependence ->InvestigateTimestep();

    VMCSolver *investigateOnebodyDensity_ChargeDensity = new VMCSolver();
    investigateOnebodyDensity_ChargeDensity ->OnebodyDensity_ChargeDensity();

*/
    return 0;


}
