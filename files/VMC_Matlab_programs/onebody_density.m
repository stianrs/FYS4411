
function onebody_density(nParticles, data, name1, name2, name3)

offset = 5000000;
x = data(offset:nParticles:end, 1);
y = data(offset:nParticles:end, 2);
z = data(offset:nParticles:end, 3);

r2 = x.^2 + y.^2 + z.^2;
r = sqrt(r2);

figure(1);
plot3(x, y, z, 'k.', 'markersize', 0.0001);
% view(0, 0);
xlabel('x');
ylabel('y');
zlabel('z');
title('Charge density');
print(name1, '-dpng', '-r300');

figure(2);
resolution = 1000;
[g x] = hist(r, resolution);
dx = diff(x(1:2));
bar(x,g/sum(g*dx));
xlabel('radial distance');
ylabel('radial electron distribution');
axis([0 6 0 1.2]);
title('Normalized Onebody distribution');
print(name2, '-dpng', '-r300');

% figure(3);
% resolution = 1000;
% [g x] = hist(r2, resolution);
% dx = diff(x(1:2));
% bar(x,g/sum(g*dx));
% xlabel('(radial distance)^2');
% ylabel('charge density distribution');
% axis([0 10 0 1]);
% title('Normalized Onebody charge density');
% print(name3, '-dpng', '-r300');

end



