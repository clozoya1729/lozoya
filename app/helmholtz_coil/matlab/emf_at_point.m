function B = emf_at_point(permeability, coordinate, current, coils)
    % EMF_AT_POINT Calculates the emf magnitude at a point in space
    B = 0;
    for i=1:3
        B = B + coils(i).emf(coordinate(i), current(i), permeability);
    end
end

