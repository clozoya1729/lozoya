function NodeTSDataConcatenate() {
    this.title = "Data | Concatenate";
    this.properties = {};
}

var nodes = [
    [NodeTSDataConcatenate, 'Concatenate']
];
var nodes = [
    NodeTSDataConcatenate,
];
var names = [
    'Concatenate',
];


nodes.forEach(
    function (node, index) {
        set_up_node(
            node[0],
            {
                color: 'rgb(75,50,30)',
                size: [225, 35],
                family: 'Data',
                objType: node[1],
                inputPorts: ['In 1', 'In 2'],
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
                color: 'rgb(75,50,30)',
                size: [225, 35],
                family: 'Data',
                objType: node.title,
                input: ['In 1', 'In 2'],
                output: ['Out'],
            },
        );
    }
);


nodes.forEach(
    function (node, index) {
        set_up_node(
            node,
            [225, 35],
            'rgb(75,50,30)',
            names[index],
            'Data',
            ['Input 1', 'Input 2'],
            ['Output'],
        );
    }
);
