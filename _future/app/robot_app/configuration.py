armLinkLengths = \
    [
        11.31,
        39.04,
        130.05,
        110.36
    ]
armJointoffsetsLength = \
    [
        0,
        -15.57,
        26.11,
        -29.12,
    ]
axes = \
    [
        'z',
        'y',
        'y',
    ]
angles = \
    [
        45,  # motor 0
        0,  # motor 1
        0,  # motor 2
    ]
armMotorOffsetsAngle = \
    [
        0,
        90,
        angles[1] + 180
    ]
motorLabels = \
    [
        'Origin',
        'Motor 0',
        'Motor 1',
        'Motor 2',
        'End Effector'
    ]
lengths = \
    [
        armLinkLengths[0],
        armLinkLengths[1],
        0,
        armLinkLengths[2],
        0,
        armLinkLengths[3]
    ]
offsets = \
    [
        armJointoffsetsLength[0],
        armJointoffsetsLength[1],
        armJointoffsetsLength[2],
        0,
        armJointoffsetsLength[3],
        0
    ]
labels = \
    [
        True,
        True,
        False,
        False,
        True,
        False,
        True,
        False,
        False,
        True
    ]
axes2 = \
    [
        'z',
        axes[0],
        'y',
        axes[1],
        'y',
        axes[2],
    ]
angles2 = \
    [
        0,
        angles[0] + armMotorOffsetsAngle[0],  # motor 0
        0,
        angles[1] + armMotorOffsetsAngle[1],  # motor 1
        0,
        angles[2] + armMotorOffsetsAngle[2],  # motor 2
    ]
motorLabels2 = \
    [
        motorLabels[0],
        motorLabels[1],
        '',
        '',
        motorLabels[2],
        '',
        motorLabels[3],
        '',
        '',
        motorLabels[4],
    ]
