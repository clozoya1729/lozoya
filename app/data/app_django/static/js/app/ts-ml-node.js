var nodes = [
    [NodeTSRegressionLinear, 'Linear'],
    [NodeTSRegressionGP, 'Gaussian Process'],
    [NodeTSRegressionKNN, 'K Nearest Neighbors'],
    [NodeTSRegressionRF, 'Random Forest'],
    [NodeTSRegressionSVM, 'Support Vector Machine'],
];
var nodes = [
    NodeTSRegressionLinear,
    NodeTSRegressionGP,
    NodeTSRegressionKNN,
    NodeTSRegressionRF,
    NodeTSRegressionSVM,
];
var nodes = [
    NodeTSRegressionLinear,
    NodeTSRegressionGP,
    NodeTSRegressionKNN,
    NodeTSRegressionRF,
    NodeTSRegressionSVM,
];
var names = [
    'Linear',
    'Gaussian Process',
    'K Nearest Neighbors',
    'Random Forest',
    'Support Vector Machine',
];
nodes.forEach(
    function (node, index) {
        set_up_node(
            node[0],
            {
                color: 'rgb(30,75,30)',
                size: [275, 35],
                family: 'Regression',
                objType: node[1],
                inputPorts: ['In'],
                outputPorts: ['Out'],
            },
        );
    }
);


nodes.forEach(
    function (node, index) {
        set_up_node(
            node,
            {
                color: 'rgb(30,75,30)',
                size: [275, 35],
                family: 'ML',
                objType: node.title,
                input: ['In'],
                output: ['Out'],
            },
        );
    }
);


nodes.forEach(
    function (node, index) {
        set_up_node(
            node,
            [275, 35],
            'rgb(30,75,30)',
            names[index],
            'ML',
            ['Input'],
            ['Output'],
        );
    }
);

function NodeTSRegressionLinear() {
    this.title = "Linear";
    this.properties = {
        'Intercept': true,
    };
}

function NodeTSRegressionGP() {
    this.title = "Gaussian Process";
    this.properties = {
        'Minimum': 0,
        'Maximum': 1,
        'Rows': 2,
        'Columns': 2,
    };
}

function NodeTSRegressionKNN() {
    this.title = "K Nearest Neighbors";
    this.properties = {
        'Mean': 0,
        'Standard Deviation': 1,
        'Rows': 2,
        'Columns': 2,
    };
}

function NodeTSRegressionRF() {
    this.title = "Random Forest";
    this.properties = {
        'Mean': 0,
        'Standard Deviation': 1,
        'Rows': 2,
        'Columns': 2,
    };
}

function NodeTSRegressionSVM() {
    this.title = "Support Vector Machine";
    this.properties = {surname: "smith"};
    this.addWidget("text", "Surname", "", {property: "surname"});
    this.addWidget("slider", "Slider", 0.5, function (value, widget, node) { /* do something with the value */
        alert(value);
    }, {min: 0, max: 1});
}

function NodeTSRegressionLinear() {
    this.title = "Regression | Linear";
    this.properties = {
        'Intercept': true,
    };
}

function NodeTSRegressionGP() {
    this.title = "Regression | Gaussian Process";
    this.properties = {
        'Minimum': 0,
        'Maximum': 1,
        'Rows': 2,
        'Columns': 2,
    };
}

function NodeTSRegressionKNN() {
    this.title = "Regression | K Nearest Neighbors";
    this.properties = {
        'Mean': 0,
        'Standard Deviation': 1,
        'Rows': 2,
        'Columns': 2,
    };
}

function NodeTSRegressionRF() {
    this.title = "Regression | Random Forest";
    this.properties = {
        'Mean': 0,
        'Standard Deviation': 1,
        'Rows': 2,
        'Columns': 2,
    };
}

function NodeTSRegressionSVM() {
    this.title = "Regression | Support Vector Machine";
    this.properties = {surname: "smith"};
    this.addWidget("text", "Surname", "", {property: "surname"});
    this.addWidget("slider", "Slider", 0.5, function (value, widget, node) { /* do something with the value */
        alert(value);
    }, {min: 0, max: 1});
}
