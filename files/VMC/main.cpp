#include "vmcsolver.h"

#include <iostream>

using namespace std;

int main()
{

/*
    VMCSolver *solver = new VMCSolver();
    solver->runMonteCarloIntegration();

    VMCSolver *investigateAlpha = new VMCSolver();
    investigateAlpha->InvestigateOptimalAlpha();

    VMCSolver *investigateBeta = new VMCSolver();
    investigateBeta->InvestigateOptimalBeta();

    VMCSolver *investigateParameters = new VMCSolver();
    investigateParameters->InvestigateOptimalParameters();
*/

    VMCSolver *investigateVariance = new VMCSolver();
    investigateVariance->InvestigateVarianceNcycles();

    return 0;
}
