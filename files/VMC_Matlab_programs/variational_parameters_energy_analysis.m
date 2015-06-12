clf;

f = load('Alpha_Energy.dat');
g = load('Beta_Energy.dat');
h = load('Parameter_Energy_Beryllium.dat');
k = load('Variance_nSampels.dat');
k2 = load('Variance_nSampels_no_imp.dat');
k3 = load('Variance_nSampels_imp.dat');
resolution = 19;

alpha = f(:,1);
alphaEnergy = f(:,2);

beta = g(:,1);
betaEnergy = g(:,2);

alphaParameter = h(:, 1)
betaParameter = h(:, 2)
parameterEnergy = h(:,3);
parameterVariance = h(:,4);

nCycles = k(:,1);
variance = k(:,2);

nCycles_no_imp = k2(:,1);
variance_no_imp = k2(:,2);
energy_no_imp = k2(:,3);

nCycles_imp = k3(:,1);
variance_imp = k3(:,2);
energy_imp = k3(:,3);


figure(1);
plot(alpha, alphaEnergy, 'b-')
xlabel('Alpha');
ylabel('Energy');
title('Ground state energy as a function of variational parameter Alpha');

print('Alpha_Energy_plot', '-dpng', '-r300');



parameterMatrix = zeros(resolution, resolution);

counter = 0;
for i=1:resolution
    for j=1: resolution
        counter = counter+1;
        parameterMatrix(i,j) = parameterEnergy(counter);
    end
end

figure(3);

xi=linspace(min(alphaParameter),max(alphaParameter),resolution);
yi=linspace(min(betaParameter),max(betaParameter),resolution);

[XI YI]=meshgrid(xi,yi);

ZI = griddata(alphaParameter, betaParameter, parameterEnergy, XI, YI);
surf(XI,YI,ZI);
title('Energy as a function of variational parameters');
xlabel('Alpha');
ylabel('Beta');
zlabel('Energy');
%print('parameter_mesh', '-dpng', '-r300');

figure (4);
plot(nCycles, variance, 'b-')
xlabel('nCycles');
ylabel('Variance');
title('Variance as a function of nCycles');
%print('variance_nCycles', '-dpng', '-r300');



figure (5);
plot(nCycles_no_imp, variance_no_imp, 'b-', nCycles_imp, variance_imp, 'r-')
legend('regular', 'importance sampling')
xlabel('nCycles');
ylabel('Variance');
title('Variance as a function of nCycles');
print('energy_nCycles_compare3 , ', '-dpng', '-r300');
