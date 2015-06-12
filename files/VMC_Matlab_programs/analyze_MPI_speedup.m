
core1_data = load('cpu_time_MPI_1.dat');
core2_data = load('cpu_time_MPI_2_new.dat');
core4_data = load('cpu_time_MPI_4.dat');
core8_data = load('cpu_time_MPI_8.dat');

ncycles = core1_data(:, 1);
core1 = core1_data(:, 2);
core2 = core2_data(:, 2);
core4 = core4_data(:, 2);
core8 = core8_data(:, 2);

speedup1 = core1./core1;
speedup2 = core1./core2;
speedup4 = core1./core4;
speedup8 = core1./core8;

speedupDev1 = 100*(1.0 - (speedup1./1));
speedupDev2 = 100*(1.0 - (speedup2./2));
speedupDev4 = 100*(1.0 - (speedup4./4));
speedupDev8 = 100*(1.0 - (speedup8./8));

figure(1);
hold on
plot(ncycles, speedup1, 'b.-');
plot(ncycles, speedup2, 'r.-');
plot(ncycles, speedup4, 'c.-');
plot(ncycles, speedup8, 'g.-');

title('Relative speedup compared by using 1 CPU');
legend('1 CPU', '2 CPU', '4 CPU', '8 CPU');
xlabel('nCycles');
ylabel('relative speedup');
print('relative_speedup', '-dpng', '-r300');
hold off


figure(2);
hold on
plot(ncycles, speedupDev1, 'b.-');
plot(ncycles, speedupDev2, 'r.-');
plot(ncycles, speedupDev4, 'c.-');
plot(ncycles, speedupDev8, 'g.-');

title('% deviation from optimal speedup');
legend('1 CPU', '2 CPU', '4 CPU', '8 CPU');
xlabel('nCycles');
ylabel('% deviation');
print('deviation_speedup', '-dpng', '-r300');
hold off
