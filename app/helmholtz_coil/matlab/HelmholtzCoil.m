classdef HelmholtzCoil
    %HELMHOLTZCOIL class
    properties
        length % meter
        spacing % meter
        windings % quantity
    end
    
    methods
        function obj = HelmholtzCoil(length, spacing, windings)
            %HELMHOLTZCOIL Construct an instance of this class
            obj.length = length;
            obj.spacing = spacing;
            obj.windings = windings;
        end
        
        function B = emf(obj, distance, current, permeability)
            % Compute the magnitude of the EMF
            K = permeability*obj.windings*current*(obj.length^2)/2;
            Q0 = ((obj.length^2)+((distance+obj.spacing/2)^2))^(3/2);
            Q1 = ((obj.length^2)+((distance-obj.spacing/2)^2))^(3/2);
            B = K*((1/Q0)+(1/Q1));
        end
    end
end

