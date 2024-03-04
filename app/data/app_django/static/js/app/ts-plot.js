function refresh_columns(field) {
    $.ajax({
        url: '/ml/ajax/refresh_columns/',
        data: {
            'x': JSON.stringify($("#" + field).val()),
            'y': JSON.stringify($("#" + field.replace(/x/g, 'y')).val()),
            'n': get_n(field),
        },
        dataType: 'json',
        success: function (data) {
            $("#" + data.field).html(data.newForm);
        }
    });
}

function update_dimensions(id) {
    var n = get_n(id);
    var selector = "select[id^='axes-selector-']select[id$='" + n + "']";
    $.ajax({
        url: '/plotter/ajax/update_dimensions/',
        data: {
            'newDim': $("#" + id).val(),
            'oldDim': $(selector).length,
            'selected': JSON.stringify(get_select_selected(selector)),
            'n': n,
        },
        dataType: 'json',
        success: function (data) {
            $("#" + data.axArea).html(data.newForm);
        }
    });
}

function update_axes(id) {

}

function refresh_columns(appType, field) {
    $.ajax({
        url: '/ml/ajax/refresh_columns/',
        data: {
            'appType': appType,
            'x': JSON.stringify($("#" + field).val()),
            'y': JSON.stringify($("#" + field.replace(/x/g, 'y')).val()),
            'n': get_n(field),
        },
        dataType: 'json',
        success: function (data) {
            $("#" + data.field).html(data.newForm);
        }
    });
}

$("#refresh_plot").click(function () {
    $.ajax({
        url: '/plotter/ajax/refresh_plot/',
        data: {
            'x': JSON.stringify($("#id_x").val()),
            'y': $("#id_y").val(),
        },
        dataType: 'json',
        success: function (data) {
            $("#ts-plot-1").html(data.plot);
            $("#ts-plot-2").html(data.histogram);
        }
    });
});

function add_plot(app) {
    $.ajax({
        url: '/plotter/ajax/add_plot/',
        data: {
            'appN': get_n(app),
            'plotN': $('.ts-plot').length + 1,
        },
        dataType: 'json',
        success: function (data) {
            $("#" + data.appArea).append(data.newPlot);
        }
    });
}

function update_dimensions(appType, field) {
    var n = get_n(field);
    $.ajax({
        url: '/plotter/ajax/update_dimensions/',
        data: {
            'appType': appType,
            'n': n,
            'newDim': $("#" + field).val(),
        },
        dataType: 'json',
        success: function (data) {
            $("#" + data.axArea).html(data.newForm);
        }
    });
}

function update_axes(appType, field) {
    var n = get_n(field);
    console.log(field);
    var xCol, yCol;
    if (field.includes('x', 2)) {
        xCol = field;
        yCol = replace_occurrence(xCol, /x/g, 2, 'y');
    } else {
        yCol = field;
        xCol = field.replace(/y/g, 'x');
    }
    $.ajax({
        url: '/plotter/ajax/update_axes/',
        data: {
            'appType': appType,
            'n': n,
            'xCol': $('#' + xCol).val(),
            'yCol': $('#' + yCol).val(),
        },
        dataType: 'json',
        success: function (data) {
            $("#" + data.field).replaceWith(data.plot);
        }
    });
}

function resize_plots(n) {
    var container = document.getElementById('plot-container-' + n);
    var q = 0;
    if (container.childNodes.length === 3) {
        q = 1;
    }
    let resizeObserver = new ResizeObserver(() => {
        container.childNodes[q].style.height = container.offsetHeight + 'px';
        resizeObserver.unobserve(container);
    });
    if (container.offsetHeight !== 0) {
        container.offsetHeight = 0;
        container.childNodes[q].style.height = '0px';
    }
    resizeObserver.observe(container);
    var d3 = Plotly.d3;
    var gd3 = d3.select('.plotly-graph-div').style({
        width: '100%',
    });
    var gd = gd3.node();
    Plotly.Plots.resize(gd);
}


$(window).resize(function (e) {
    $("model-plot-container").each(function () {
        var container = document.getElementById($(this).id);
        var q = 0;
        if (container.childNodes.length === 3) {
            q = 1;
        }
        container.style.height = '60vh';
        container.childNodes[q].style.height = container.offsetHeight + 'px';
    });
    var d3 = Plotly.d3;
    var gd3 = d3.select('.plotly-graph-div').style({
        width: '100%',
    });
    var gd = gd3.node();
    Plotly.Plots.resize(gd);
});

function refresh_columns(objectType, field) {
    $.ajax({
        url: '/ml/ajax/refresh_columns/',
        data: {
            'objectType': objectType,
            'x': JSON.stringify($("#" + field).val()),
            'y': JSON.stringify($("#" + field.replace(/x/g, 'y')).val()),
            'n': get_n(field),
        },
        dataType: 'json',
        success: function (data) {
            $("#" + data.field).html(data.newForm);
        }
    });
}

function update_dimensions(objectType, field) {
    var n = get_n(field);
    $.ajax({
        url: '/plotter/ajax/update_dimensions/',
        data: {
            'objectType': objectType,
            'n': n,
            'newDim': $("#" + field).val(),
        },
        dataType: 'json',
        success: function (data) {
            $("#" + data.axArea).html(data.newForm);
        }
    });
}

function update_axes(objectType, field) {
    var n = get_n(field);
    console.log(field);
    var xCol, yCol;
    if (field.includes('x', 2)) {
        xCol = field;
        yCol = replace_occurrence(xCol, /x/g, 2, 'y');
    } else {
        yCol = field;
        xCol = field.replace(/y/g, 'x');
    }
    $.ajax({
        url: '/plotter/ajax/update_axes/',
        data: {
            'objectType': objectType,
            'n': n,
            'xCol': $('#' + xCol).val(),
            'yCol': $('#' + yCol).val(),
        },
        dataType: 'json',
        success: function (data) {
            $("#" + data.field).replaceWith(data.plot);
        }
    });
}

