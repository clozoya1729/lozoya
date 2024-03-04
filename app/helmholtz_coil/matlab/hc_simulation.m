%hold on;
syms x y z t;
%[x0,y0,z0] = meshgrid(linspace(-2,2,5), linspace(-2,2,5), linspace(-2,2,5));
turns = 15;
spacePermeability = 4*pi*10^-7; % (Tesla * meter / Amp)
current = 3.6; % (Amp)
length = 3; % Side length of square (meter)
spacing = 1.6; % Spacing between coils (meter)

%[Bx By Bz] = get_vector_potentials(x, y, z);
Bz = magnetic_field(z, turns, spacePermeability, current, spacing, length);
Bz = Bz * 10000; % convert from (Tesla) to (Gauss)
k = 200;
scale = 50;
a = zeros(1,k-1);
b = zeros(1,k-1);
disp(b);
for i = 1:k-1
    a(i) = (i-k/2)/scale
    b(i) = magnetic_field((i-k/2)/scale, turns, spacePermeability, current, spacing, length)*10000;
end
disp(b);
plot(a, b);
%fplot(Bz);
%figure(2);
%quiver3(x0, y0, z0, Bx, By, Bz);
%ylim([0,2] );
xlabel('z (m)');
ylabel('Bz (Gauss)');
%hold off;
%grid on;

function [Bx, By, Bz] = get_vector_potentials(x, y, z)
    V(x) = x./x;
    Z(x) = x*0;
    Bx = vectorPotential([V; Z; Z], [x y z]);
    By = vectorPotential([Z; V; Z], [x y z]);
    Bz = vectorPotential([Z; Z; V], [x y z]);
end

function Bx = magnetic_field(x, turns, spacePermeability, current, spacing, length)
    numerator = turns * spacePermeability * current * length^2;
    denominator1 = 2*pi*(((spacing/2)-x)^2+(length/2)^2)^(3/2);
    denominator2 = 2*pi*(((spacing/2)+x)^2+(length/2)^2)^(3/2);
    %Bx = numerator/(denominator1+denominator2);
    Bx = (numerator/denominator1)+(numerator/denominator2);
end