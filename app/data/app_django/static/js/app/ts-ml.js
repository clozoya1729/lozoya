$("#ts-model-settings-1").change(function () {
    update_controls();
});

function update_controls() {
    var model = $("#ts-model-settings-1").val();
    var modelSelect = $("#ts-model-settings-1 option:selected").text();
    $.ajax({
        url: '/ml/ajax/refresh_controls/',
        data: {
            'model': model
        },
        dataType: 'json',
        success: function (data) {
            $("#id_model_settings").html(data.settings);
            $("#model_settings_legend").html(modelSelect + ' Hyperparameters');
            if (model.includes('SupportVectorMachine')) {
                svm_kernel();
            }
        }
    });
}

function update_model(field) {
    $.ajax({
        url: '/ml/ajax/update_model/',
        data: {
            'n': get_n(field),
            'field': field,
            'value': validate_parameter(field),
        },
        dataType: 'json',
    });
}

function fit_model(app) {
    $.ajax({
        url: 'ajax/fit_model/',
        data: {
            'n': get_n(app),
        },
        dataType: 'json',
        success: function (data) {
            if (data.errors.length === 0) {
                $("#" + data.plotID).html(data.plot);
                $("#" + data.histogramID).html(data.histogram);
                download_button('show');
            } else {
                alert(data.errors);
            }
        }
    });
}

$("#ts-model-settings-1").change(function () {
    update_controls();
});

function update_controls() {
    var model = $("#ts-model-settings-1").val();
    var modelSelect = $("#ts-model-settings-1 option:selected").text();
    $.ajax({
        url: '/ml/ajax/refresh_controls/',
        data: {
            'model': model
        },
        dataType: 'json',
        success: function (data) {
            $("#id_model_settings").html(data.settings);
            $("#model_settings_legend").html(modelSelect);
            if (model.includes('SupportVectorMachine')) {
                svm_kernel();
            }
        }
    });
}

$("select[id*='model'][multiple]:not('[id*=settings]') > option").mousedown(function (e) {
    switch (e.which) {
        case 1:
            e.preventDefault();
            $(this).prop('selected', !$(this).prop('selected'));
            update_model($(this).parent().attr('id'));
            return false;
    }

});

function update_controls(appType, n) {
    var model = $("#id_model_settings_select option:selected");
    $.ajax({
        url: '/ml/ajax/update_controls/',
        data: {
            'appType': appType,
            'model': model.val(),
            'n': n,
        },
        dataType: 'json',
        success: function (data) {
            $("#id_model_settings_legend").html(model.text());
            $("#id_model_settings_form").html(data.settings);
            if (model.val() !== undefined) {
                if (model.val().includes('SupportVectorMachine')) {
                    svm_kernel();
                }
            }
        }
    });
}

function update_model(appType, field) {
    $.ajax({
        url: '/ml/ajax/update_model/',
        data: {
            'appType': appType,
            'n': get_n(field),
            'field': field,
            'value': validate_parameter(field),
        },
        dataType: 'json',
    });
}

function fit_model(app) {
    $.ajax({
        url: '/ml/ajax/fit_model/',
        data: {
            'n': get_n(app),
        },
        dataType: 'json',
        success: function (data) {
            if (data.errors.length === 0) {
                $("#" + data.plotID).html(data.plot);
                // download_button('show');
            } else {
                alert(data.errors);
            }
        }
    });
}

function validate_parameter(field) {
    var f = $("#" + field);
    var value;
    if (f.is(':checkbox')) {
        value = f.is(':checked');
    } else {
        value = f.val();
        var max = f.attr('max');
        if (max !== undefined) {
            if (value > parseFloat(max)) {
                value = max;
                f.val(max);
            }
        }
        var min = f.attr('min');
        if (min !== undefined) {
            if (value < parseFloat(min)) {
                value = min;
                f.val(min);
            }
        }
    }
    return JSON.stringify(value);
}

function display_field(field, m) {
    if (m === 'show') {
        $("#" + field).show();
        $("[for=" + field + "]").show()
    }
    if (m === 'hide') {
        $("#" + field).hide();
        $("[for=" + field + "]").hide()
    }
}

function update_controls(objectType, n) {
    var model = $("#id_model_settings_select option:selected");
    $.ajax({
        url: '/ml/ajax/update_controls/',
        data: {
            'objectType': objectType,
            'model': model.val(),
            'n': n,
        },
        dataType: 'json',
        success: function (data) {
            $("#id_model_settings_legend").html(model.text());
            $("#id_model_settings_form").html(data.settings);
            if (model.val() !== undefined) {
                if (model.val().includes('SupportVectorMachine')) {
                    svm_kernel();
                }
            }
        }
    });
}

function update_model(objectType, field) {
    $.ajax({
        url: '/ml/ajax/update_model/',
        data: {
            'objectType': objectType,
            'n': get_n(field),
            'field': field,
            'value': validate_parameter(field),
        },
        dataType: 'json',
    });
}

function svm_kernel(id) {
    var val = $("#" + id).val();
    var n = get_n(id);
    var degree = 'svmDegree-' + n;
    var coef0 = 'svmCoef0-' + n;
    var gamma = 'svmGamma-' + n;
    if (val !== 'poly') {
        display_field(degree, 'hide');
    } else {
        display_field(degree, 'show');
    }
    if (val !== 'poly' && val !== 'sigmoid') {
        display_field(coef0, 'hide');
    } else {
        display_field(coef0, 'show');
    }
    if (val === 'linear') {
        display_field(gamma, 'hide');
    } else {
        display_field(gamma, 'show');
    }
}

function knn_metric(id) {
    var val = $("#" + id).val();
    var n = get_n(id);
    var p = 'knnP-' + n;
    if (val !== 'minkowski') {
        display_field(p, 'hide');
    } else {
        display_field(p, 'show');
    }
}

$(document).ready(function () {
    var choice = 'GaussianProcessRegression';
    $("#ts-model-settings-1").val(choice);
    update_controls();
});

