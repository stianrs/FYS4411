#ifndef INVESTIGATE_H
#define INVESTIGATE_H


double InvestigateOptimalStep();
void InvestigateOptimalAlpha();
void InvestigateOptimalParameters();
void InvestigateVarianceNcycles();
void InvestigateCPUtime(int my_rank, int world_size);
void InvestigateTimestep();
void BlockingFunc();
void OnebodyDensity_ChargeDensity();


#endif // INVESTIGATE_H


