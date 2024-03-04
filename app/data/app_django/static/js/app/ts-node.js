'use strict';

function add_ports(node, metadata) {
    metadata.input.forEach(function (element, index) {
        node.prototype.addInput(element, 'string')
    });
    metadata.output.forEach(function (element, index) {
        node.prototype.addOutput(element, 'string')
    });
}

function add_ports(node, input, output) {
    input.forEach(function (element, index) {
        node.prototype.addInput(element, 'string')
    });
    output.forEach(function (element, index) {
        node.prototype.addOutput(element, 'string')
    });
}

function add_ports(node, metadata) {
    metadata.inputPorts.forEach(function (element, index) {
        node.prototype.addInput(element, 'string')
    });
    metadata.outputPorts.forEach(function (element, index) {
        node.prototype.addOutput(element, 'string')
    });
}

function format_node(node, size, title, color) {
    node.size = size;
    node.title = title;
    node.title_color = color;
}

function set_node_events(node) {
    node.prototype._disconnectInput = node.prototype.disconnectInput;
    node.prototype.disconnectInput = function (slot) {
        this._disconnectInput(slot);
        ts_update_input(node, '');
    };
    node.prototype.onAdded = function () {
        ts_create_object(node);
    };
    node.prototype.onRemoved = function () {
        ts_delete_object(node);
    };
    node.prototype.onConnectInput = function (action, data, value) {
        ts_update_input(node, value._data); //value._data is hacky??
    };
    node.prototype.onPropertyChanged = function (property, value) {
        ts_update_object(node, property, value);
    };
    node.prototype.onDblClick = function () {
        ts_set_selected_node(node);
        ts_update_sidebars(node);
    };
    node.prototype.onDeselected = function () {
        ts_set_selected_node(null);
        ts_reset_sidebars();
    }
}

function set_node_metadata(node, metadata) {
    node.title = metadata.objType;
    node.title_color = metadata.color;
    node.prototype.metadata = metadata
}

function set_up_node(node, metadata) {
    LiteGraph.registerNodeType(`${metadata.family}/${metadata.objType}`, node);
    set_node_metadata(node, metadata);
    add_ports(node, metadata);
    set_node_events(node);
}

function set_up_node(node, size, color, title, family, input, output) {
    LiteGraph.registerNodeType(family + "/" + title, node);
    format_node(node, size, title, color);
    add_ports(node, input, output);
    let urls = get_urls(family);
    node.prototype.origami = node.prototype.disconnectInput;
    node.prototype.disconnectInput = function (slot) {
        this.origami(slot);
        ts_update_input(node, urls['updateInputURL'], '');
    };
    node.prototype.onAdded = function () {
        ts_create_object(node, urls['createURL']);
    };
    node.prototype.onRemoved = function () {
        ts_delete_object(node, urls['deleteURL']);
    };
    node.prototype.onConnectInput = function (action, data, value) {
        ts_update_input(node, urls['updateInputURL'], value._data); //value._data is hacky??
    };
    node.prototype.onPropertyChanged = function (property, value) {
        ts_update_object(node, urls['updatePropertyURL'], property, value);
    };
    node.prototype.onDblClick = function () {
        $('#myModal').show();
    }
}

function set_up_node(node, metadata) {
    LiteGraph.registerNodeType(`${metadata.family}/${metadata.objType}`, node);
    set_node_metadata(node, metadata);
    add_ports(node, metadata);
    node.prototype._disconnectInput = node.prototype.disconnectInput;
    node.prototype.disconnectInput = function (slot) {
        this._disconnectInput(slot);
        ts_update_input(node, '');
    };
    node.prototype.onAdded = function () {
        ts_create_object(node);
    };
    node.prototype.onRemoved = function () {
        ts_delete_object(node);
    };
    node.prototype.onConnectInput = function (action, data, value) {
        ts_update_input(node, value._data); //value._data is hacky??
    };
    node.prototype.onPropertyChanged = function (property, value) {
        ts_update_object(node, property, value);
    };
    node.prototype.onSelected = function () {
        ts_update_sidebars(node);
    };
    node.prototype.onDeselected = function () {
        ts_reset_sidebars();
    }
}

function set_node_metadata(node, metadata) {
    node.size = metadata.size;
    node.title = metadata.objType;
    node.title_color = metadata.color;
    node.prototype.metadata = metadata
}

$(document).ready(//disable double click menu
    function () {
        LGraphCanvas.prototype.showShowNodePanel = function () {
        };
    });
