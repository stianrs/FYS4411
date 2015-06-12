
block = load('Blocking_data_neon.dat');
energy = block(:, 1);
energySquared = block(:, 2);
data_size = length(energy);
%disp(data_size);

block_trials = 1500;

block_size_values = zeros(block_trials, 1);
block_size_variance = zeros(block_trials, 1);

for i=1:block_trials 
    %blocks = (block_trials - i+1)*data_size/(10*i);
    
    blocks = data_size/(150*i);
    energy_trial_sum = 0;
    energySquared_trial_sum = 0;
    
    for j=1:blocks
        block_size = data_size/blocks;
        
        index_start = floor(1 + (j-1)*block_size);
        index_stop = floor(j*block_size);
        
        n = (index_stop - index_start) + 1;

        energy_mean_block = sum(energy(index_start:index_stop))/n;
        %energySquared_mean_block = sum(energySquared(index_start:index_stop));
       
        energy_trial_sum = energy_trial_sum + energy_mean_block;
        %energySquared_trial_sum = energySquared_trial_sum + energySquared_mean_block;
        energySquared_trial_sum = energySquared_trial_sum + energy_mean_block*energy_mean_block;
        
        %disp(n);
        
    end
    
    energy_mean_blockSize = energy_trial_sum/blocks;
    energySquared_mean_blockSize = energySquared_trial_sum/blocks;    
    %variance_mean_blockSize = (energySquared_mean_blockSize - energy_mean_blockSize^2);
    variance_mean_blockSize = (energySquared_mean_blockSize - (energy_mean_blockSize*energy_mean_blockSize))/blocks;
    
    block_size_values(i) = floor(block_size);
    block_size_variance(i) = variance_mean_blockSize;
end
	
SD =  sqrt(block_size_variance); 

plot(block_size_values, SD, 'b-', 'linewidth', 2);
xlabel('block size');
ylabel('SD');
title('SD as a function of block size');
print('blocking_neon', '-dpng', '-r300');




