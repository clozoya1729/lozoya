function coils = make_hc3(lengths, spacings, windings)
    % MAKE_HC3 Returns a vector containing a Helmholtz Coil
    % corresponding to each axis.
    coils = HelmholtzCoil.empty(3, 0);
    for i=1:3
        coils(i) = HelmholtzCoil(lengths(i), spacings(i), windings(i));
    end
end

