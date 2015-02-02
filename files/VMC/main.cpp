#include "vmcsolver.h"

#include <iostream>

using namespace std;

int main()
{


    VMCSolver *solver = new VMCSolver();
    solver->runMonteCarloIntegration();

    VMCSolver *investigateAlpha = new VMCSolver();
    investigateAlpha->InvestigateOptimalAlpha();

    VMCSolver *investigateBeta = new VMCSolver();
    investigateBeta->InvestigateOptimalBeta();

    /*
    VMCSolver *investigateParameters = new VMCSolver();
    investigateParameters->InvestigateOptimalParameters();

    VMCSolver *investigateStep = new VMCSolver();
    investigateStep->InvestigateOptimalStep();
*/

    return 0;
}
