var names = [
    'Slope-Intercept',
    'Sine',
];
var names = [
    'Range',
    'Random Uniform',
    'Random Normal',
];
var nodes = [
    NodeTSMathSlopeIntercept,
    NodeTSMathSine,
];
var nodes = [
    NodeTSRange,
    NodeTSRandomUniform,
    NodeTSRandomNormal,
];
var nodes = [
    [NodeTSRange, 'Range'],
    [NodeTSRandomUniform, 'Random Uniform'],
    [NodeTSRandomNormal, 'Random Normal'],
];

var nodes = [
    [NodeTSMathSlopeIntercept, 'Slope-Intercept'],
    [NodeTSMathSine, 'Sine'],
];

function NodeTSMathSlopeIntercept() {
    this.size = [140, 35];
    this.properties = {
        'Slope': 1, 'Intercept': 0,
    };
}

function NodeTSMathSine() {
    this.size = [90, 35];
    this.properties = {
        'Amplitude': 1, 'Frequency': 1, 'Horizontal Offset': 0, 'Vertical Offset': 0,
    };
}

function NodeTSMathSine() {
    this.properties = {
        'Amplitude': 1, 'Frequency': 1, 'Horizontal Offset': 0, 'Vertical Offset': 0,
    };
}

function NodeTSMathSine() {
    this.title = "Math | Sin";
    this.properties = {
        'Amplitude': 1, 'Frequency': 1, 'Horizontal Offset': 0, 'Vertical Offset': 0,
    };
}

function NodeTSRandomUniform() {
    this.properties = {
        'Minimum': 0, 'Maximum': 1, 'Rows': 10, 'Columns': 1,
    };
}

function NodeTSRandomNormal() {
    this.properties = {
        'Mean': 0, 'Standard Deviation': 1, 'Rows': 10, 'Columns': 1,
    };

}

function NodeTSRange() {
    this.properties = {
        'Minimum': 0, 'Maximum': 1, 'Rows': 10, 'Columns': 1,
    };
}

function NodeTSRandomUniform() {
    this.size = [150, 35];
    this.properties = {
        'Minimum': 0, 'Maximum': 1, 'Rows': 10, 'Columns': 1,
    };

}

function NodeTSRandomNormal() {
    this.size = [150, 35];
    this.properties = {
        'Mean': 0, 'Standard Deviation': 1, 'Rows': 10, 'Columns': 1,
    };

}

function NodeTSRandomUniform() {
    this.title = "Generator | Random Uniform";
    this.properties = {
        'Minimum': 0, 'Maximum': 1, 'Rows': 2, 'Columns': 2,
    };
}

function NodeTSMathSlopeIntercept() {
    this.title = "Math | Slope-Intercept";
    this.properties = {
        'Slope': 1, 'Intercept': 0,
    };
}

function NodeTSRange() {
    this.size = [90, 35];
    this.properties = {
        'Minimum': 0, 'Maximum': 1, 'Rows': 10, 'Columns': 1,
    };
}

function NodeTSRange() {
    this.title = "Generator | Range";
    this.properties = {
        'Minimum': 0, 'Maximum': 1, 'Rows': 2, 'Columns': 2,
    };
}

function NodeTSRandomNormal() {
    this.title = "Generator | Random Normal";
    this.properties = {
        'Mean': 0, 'Standard Deviation': 1, 'Rows': 2, 'Columns': 2,
    };
}

nodes.forEach(function (node, index) {
    set_up_node(node[0], {
        color: 'rgb(30,30,75)', family: 'Math', objType: node[1], inputPorts: ['In'], outputPorts: ['Out'],
    },);
});

nodes.forEach(function (node, index) {
    set_up_node(node[0], {
        color: 'rgb(75,30,30)', family: 'Generator', objType: node[1], inputPorts: [], outputPorts: ['Out'],
    },);
});
nodes.forEach(function (node, index) {
    set_up_node(node, {
        color: 'rgb(30,30,75)', size: [150, 35], family: 'Math', objType: names[index], input: ['In'], output: ['Out'],
    },);
});
nodes.forEach(function (node, index) {
    set_up_node(node, {
        color: 'rgb(75,30,30)', size: [150, 35], family: 'Generator', objType: names[index], input: [], output: ['Out'],
    },);
});
nodes.forEach(function (node, index) {
    set_up_node(node, [225, 35], 'rgb(75,30,30)', names[index], 'Generator', [], ['Output'],);
});
nodes.forEach(function (node, index) {
    set_up_node(node, [180, 35], 'rgb(30,30,75)', names[index], 'Math', ['Input'], ['Output'],);
});
