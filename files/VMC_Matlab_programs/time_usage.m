
cpu = load('cpu_time.dat')
cpu_imp = load('cpu_time_imp.dat')

n_cpu = cpu(:, 1);
time_cpu = cpu(:, 2);

n_cpu_imp = cpu_imp(:, 1);
time_cpu_imp = cpu_imp(:, 2);

figure (1);
plot(n_cpu_imp, time_cpu, 'b-', n_cpu_imp, time_cpu_imp, 'r-')
legend('regular', 'importance sampling')
xlabel('nCycles');
ylabel('time [sec]');
title('CPU usage as a function of nCycles');
%print('cpu_compare_xxx , ', '-dpng', '-r300');






















