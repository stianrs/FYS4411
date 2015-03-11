/*
This is the main program used to run Monte Carlo simulations with "vmcsolver.cpp" and run different functions
in "investigate.cpp" using vmcsolver.cpp.
*/

#include "vmcsolver.h"
#include "investigate.h"

#include <iostream>
#include <armadillo>


using namespace std;
using namespace arma;

int main()
{

    int nCycles = 1000000;

    //VMCSolver *solver = new VMCSolver();
    //solver->runMonteCarloIntegration(nCycles);


    //VMCSolver *investigateOnebodyDensity_ChargeDensity = new VMCSolver();
    //investigateOnebodyDensity_ChargeDensity ->OnebodyDensity_ChargeDensity();

    VMCSolver *investigateCPU = new VMCSolver();
    investigateCPU->InvestigateCPUtime();

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
