#include "vmcsolver.h"

#include <iostream>

using namespace std;

int main()
{
    VMCSolver *solver = new VMCSolver();
    solver->runMonteCarloIntegration();

    VMCSolver *investigateAlpha = new VMCSolver();
    investigateAlpha->InvestigateOptimalAlpha();

    return 0;
}
