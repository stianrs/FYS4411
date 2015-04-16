#ifndef INVESTIGATE_H
#define INVESTIGATE_H


double InvestigateOptimalStep();
void InvestigateOptimalAlpha();
void InvestigateOptimalParameters(int my_rank, int world_size);
void InvestigateVarianceNcycles();
void InvestigateCPUtime(int my_rank, int world_size);
void InvestigateTimestep();
void BlockingFunc(int my_rank, int world_size);
void OnebodyDensity_ChargeDensity(int my_rank, int world_size);


#endif // INVESTIGATE_H


