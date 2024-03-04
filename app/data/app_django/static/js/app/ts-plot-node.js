function NodeTSPlotScatter2D() {
    this.size = [110, 35];
    this.properties = {};
}

function NodeTSPlotScatter2D() {
    this.title = "Plot | Scatter2D";
    this.properties = {};
}

var nodes = [
    NodeTSPlotScatter2D,
];
var names = [
    'Scatter 2D',
];
var nodes = [
    [NodeTSPlotScatter2D, 'Scatter 2D']
];


nodes.forEach(
    function (node, index) {
        set_up_node(
            node[0],
            {
                color: 'rgb(30,75,50)',
                family: 'Plot',
                objType: node[1],
                inputPorts: ['In'],
                outputPorts: [],
            },
        );
    }
);


nodes.forEach(
    function (node, index) {
        set_up_node(
            node,
            {
                color: 'rgb(30,75,50)',
                size: [110, 35],
                family: 'Plot',
                objType: names[index],
                input: ['In'],
                output: [],
            },
        );
    }
);


nodes.forEach(
    function (node, index) {
        set_up_node(
            node,
            [400, 400],
            'rgb(30,75,50)',
            names[index],
            'Plot',
            ['Input'],
            [],
        );
    }
);
