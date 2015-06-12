
timestep_data = load('timestep_dependence2.dat');

timestep = timestep_data(:, 1);
energy = timestep_data(:, 2);
r_ij = timestep_data(:, 3); 
cpu = timestep_data(:, 4);

data_size = length(energy);


figure(1);
plot(timestep, energy, 'b-')
xlabel('Timestep');
ylabel('Energy');
title('Ground state energy as a function of timestep with importance sampling');

print('Timestep_plot', '-dpng', '-r300');



figure(2);
plot(timestep, cpu, 'b-')
% xlabel('Timestep');
ylabel('Time [sec]');
title('Time as a function of timestep with importance sampling');

print('Timestep_time_plot', '-dpng', '-r300');



figure(3);
plot(timestep, r_ij, 'b-')
xlabel('Timestep');
ylabel('r_i_j');
title('Averange electron distance as a function of timestep with importance sampling');

print('Timestep_r_ij_plot', '-dpng', '-r300');






























