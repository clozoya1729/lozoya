% PARAMETERS
permeability = 10000*4*pi*10^(-7); % Gauss*meter/Amp
lengths = [2 2 2];
spacings = [1.1 1.1 1.1];
windings = [15 15 15];
current = [3.6 3.6 3.6];
resolution = 20;
scatterSize = 50;
% Grid
linspaceX = linspace(-lengths(1), lengths(1), 2*resolution);
linspaceY = linspace(-lengths(2), lengths(2), 2*resolution);
linspaceZ = linspace(-lengths(3), lengths(3), 2*resolution);
[X,Y,Z] = meshgrid(linspaceX, linspaceY, linspaceZ);
x = [X(:); X(:); X(:)];
y = [Y(:); Y(:); Y(:)];
z = [Z(:); Z(:); Z(:)];
% Helmholtz Coils
coils = make_hc3(lengths, spacings, windings);
emfVolume = emf_volume(permeability, current, coils, resolution);
% PLOT
scatter3(X(:), Y(:), Z(:), scatterSize, emfVolume(:), 'filled');
xlabel('x (m)')
ylabel('y (m)')
zlabel('z (m)')
% Colorbar
cb = colorbar;
cb.Label.String = 'B (Gauss)';

