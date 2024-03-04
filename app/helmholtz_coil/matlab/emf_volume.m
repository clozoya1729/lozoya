function emfVolume = emf_volume(permeability, current, coils, resolution)
    emfVolume = zeros(coils(1).length*resolution,coils(2).length*resolution);
    for i=1:1:coils(1).length*resolution
        for j=1:1:coils(2).length*resolution
            for k=1:1:coils(3).length*resolution
                pointX = i-0.5-resolution*coils(1).length/2;
                pointY = j-0.5-resolution*coils(2).length/2;
                pointZ = k-0.5-resolution*coils(3).length/2;
                point = [pointX pointY pointZ];
                emfVolume(i,j,k) = emf_at_point(permeability, point, current, coils);
            end
        end
    end
end

