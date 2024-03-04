def get_ambient_B():
    sensorReading = 0  # Query sensor to get ambient EMF magnitude
    return sensorReading


def get_dependent_weight(independentWeights):
    return 1 - sum(independentWeights)


def get_currents(requiredB, weights, characteristicConstants):
    currents = []
    for i in range(3):
        ambientB = get_ambient_B()
        hcB = requiredB - ambientB
        q = weights[i] / characteristicConstants[i]
        currents[i] = q * hcB
    return currents


def arrange_weights(independentWeights, dependentWeight, dependentAxis):
    weights = [None, None, None]
    weights[dependentAxis] = dependentWeight
    j = 0
    for i in range(3):
        if i != dependentAxis:
            weights[i] = independentWeights[j]
            j += 1
    return weights


def run_controller(requiredB, independentWeights, dependentAxis, characteristicConstants):
    '''
    requiredB: Specified by user.
    independentWeights: Specified by user.
    dependentAxis:  Specified by user.
    characteristicConstants: Determined from Helmholtz Coil design.
    '''
    dependentWeight = get_dependent_weight(independentWeights)
    weights = arrange_weights(independentWeights, dependentWeight, dependentAxis)
    currents = get_currents(requiredB, weights, characteristicConstants)
    return currents

