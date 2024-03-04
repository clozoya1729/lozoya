'use strict';
const tsDashboardID = '#dashboardID';
const tsSidebarHeadingLeftID = '#ts-sidebar-heading-left';
const tsSidebarHeadingRightID = '#ts-sidebar-heading-right';
const tsSidebarLeftUpperID = '#ts-sidebar-left-upper';
const tsSidebarLeftID = '#ts-sidebar-left';
const tsSidebarRightID = '#ts-sidebar-right';
const tsSidebarRightUpperID = '#ts-sidebar-right-upper';
const tsSidebarLeftLowerID = '#ts-sidebar-left-lower';
const tsSidebarRightLowerID = '#ts-sidebar-right-lower';
const sidebarsObj = {
    'leftUpper': tsSidebarLeftUpperID,
    'leftLower': tsSidebarLeftLowerID,
    'rightUpper': tsSidebarRightUpperID,
    'rightLower': tsSidebarRightLowerID,
};
const urlPrefix = '/ts_apps';
const urls = {
    'create': urlPrefix + '/ajax/create_object',
    'delete': urlPrefix + '/ajax/delete_object',
    'updateSidebars': urlPrefix + '/ajax/update_sidebars',
    'updateInput': urlPrefix + '/ajax/update_object_input',
    'updateProperty': urlPrefix + '/ajax/update_object_property',
};
let _focused_node = null;

function add_app() {
    /*
    Adds an app to the document and the page.
     */
    $.ajax({
        url: '/general/ajax/add_app/', data: {}, dataType: 'json', success: function (data) {
            if (data.error === '') {
                $("#" + data.appArea).append(data.newApp);
                protect('sidebar-' + data.n);
                sidebar_click('sidebar-' + data.n);
                protect('app-sidebar-' + data.n);
                sidebar_click('app-sidebar-' + data.n);
            } else {
                alert(data.error);
            }
        }
    });
}

function get_children(parent) {
    return [].slice.call(document.getElementById(parent).getElementsByTagName('*'));
}

function get_d(field) {
    return get__(field, 2);
}

function get_n(field) {
    return get__(field, 1);
}

function get__(field, i) {
    var s = field.split('-');
    return s[s.length - i];
}

function get_select_selected(selector) {
    return $(selector + " option:selected").map(function () {
        return this.value;
    }).get();
}

function get_urls(family) {
    let urlPrefix = '/ts_' + family.toLowerCase()

    return {
        'urlPrefix': urlPrefix,
        'createURL': urlPrefix + '/ajax/create_object',
        'deleteURL': urlPrefix + '/ajax/delete_object',
        'updateInputURL': urlPrefix + '/ajax/update_input',
        'updatePropertyURL': urlPrefix + '/ajax/update_object',
    };
}

function insert_app(objectType) {
    objectType = objectType.toLowerCase();
    alert('Inserted ' + objectType + ' app.');
}

function jax_call(node, url, data, success) {
    $.ajax({
        url: url, data: data, async: false, dataType: 'json', success: function (data) {
            success(data);
        },
    })
}

function jax_call(node, url, extraData, success) {
    $('html').css('cursor', 'progress');
    let baseData = {
        'dashboardID': $(tsDashboardID).text(), 'node': JSON.stringify(node.prototype.metadata),
    };
    $.ajax({
        url: url, data: Object.assign(baseData, extraData), dataType: 'json', success: function (data) {
            success(data);
            $('html').css('cursor', 'auto');
        },
    })
}

function jax_call(node, url, extraData, success) {
    let baseData = {
        'dashboardID': $(tsDashboardID).text(), 'node': JSON.stringify(node.prototype.metadata),
    };
    $.ajax({
        url: url, data: Object.assign(baseData, extraData), async: false, dataType: 'json', success: function (data) {
            success(data);
        },
    })
}

function JIntegral() {
    $("select[id*='ts-model-x'][multiple] > option").mousedown(JIntegral_x);
    $("select[id*='ts-model-algorithms'][multiple] > option").mousedown(JIntegral_x);
}

function JIntegral_x(e) {
    switch (e.which) {
        case 1:
            e.preventDefault();
            $(this).prop('selected', !$(this).prop('selected'));
            update_model($(this).parent().attr('objectType'), $(this).parent().attr('id'));
            return false;
    }
}

function remove_app(selector) {
    /*
    Removes an app from the database and from the page.
     */
    var n = get_n(selector);
    if (confirm('Remove app?')) {
        $.ajax({
            url: '/general/ajax/remove_app/', data: {
                'dashboardID': $('#dashboard-id').val(), 'n': n,
            }, dataType: 'json', success: function (data) {
                close_sidebar('app-sidebar-' + n);
                $("#app-zone-" + n).remove();
                renumber_app_headers(data.appNames);
                renumber(data.oldIDList);
            }
        });
    }
}

function renumber(oldIDList) {
    for (var i = 0; i < oldIDList.length; i++) {
        var fieldID = oldIDList[i];
        var oldN = get_n(fieldID);
        var newN = Number(oldN - 1);
        var newID = fieldID.slice(0, fieldID.length - String(newN).length) + newN;
        replace_events("onclick", fieldID, newID);
        replace_events("onchange", fieldID, newID);
        replace_property("href", fieldID, "app-accordion-" + oldN, "app-accordion-" + newN);
        replace_property("data-parent", fieldID, "app-accordion-" + oldN, "app-accordion-" + newN);
        $("#" + fieldID).prop("id", newID);
    }
}

function renumber_app_headers(appNames) {
    var regExp = /{([^}]+)}/;
    for (var i = 0; i < appNames.length; i++) {
        var n = appNames[i];
        var r = new RegExp('app-replace-' + n, "g");
        var butt = $("#" + 'remove-button-' + (n));
        var event = butt.prop('onclick');
        if (event != null) {
            event = regExp.exec(event.toString())[1];
            butt.attr('onclick', event.replace(r, 'app-replace-' + (n - 1)).trim());
        }
        var os = 'options-' + (n - 1);
        var ss = 'settings-' + (n - 1);
        var nos = 'options-' + n;
        var nss = 'settings-' + n;
        var a = $("#" + 'options-button-' + n);
        var b = $("#" + 'settings-button-' + n);
        var c = $("#" + 'close-options-' + n);
        var d = $("#" + 'close-settings-' + n);
        r = new RegExp(nos, "g");
        event = regExp.exec(a.prop('onclick').toString())[1];
        a.attr('onclick', event.replace(r, os).trim());
        event = regExp.exec(c.prop('onclick').toString())[1];
        c.attr('onclick', event.replace(r, os).trim());
        r = new RegExp(nss, "g");
        event = regExp.exec(b.prop('onclick').toString())[1];
        b.attr('onclick', event.replace(r, ss).trim());
        event = regExp.exec(d.prop('onclick').toString())[1];
        d.attr('onclick', event.replace(r, ss).trim());
        var oldHeader = "App " + n;
        var newHeader = "App " + (n - 1);
        var x = $("#app-heading-" + (n));
        x.html(x.html().replace(oldHeader, newHeader));
        x = $("#options-title-" + (n));
        x.html(x.html().replace(oldHeader, newHeader));
        x = $("#settings-title-" + (n));
        x.html(x.html().replace(oldHeader, newHeader));
        renumber_sidebar_contents("options-body-" + n, n);
        renumber_sidebar_contents("settings-body-" + n, n);

    }
}

function renumber_sidebar_contents(field, n) {
    var children = get_children(field);
    for (var i = 0; i < children.length; ++i) {
        var id_ = children[i].id;
        if (id_ !== '') {
            var thing = $('#' + id_);
            var newID = id_.slice(0, id_.length - String(n - 1).length) + (n - 1);
            if (thing.prop('onclick') !== null) {
                replace_events("onclick", id_, newID);
            }
            if (thing.prop('onchange') !== null) {
                replace_events("onchange", id_, newID);
            }
            if (thing.prop('onfocus') !== null) {
                replace_events("onfocus", id_, newID);
            }
            var label = $("label[for='" + thing.attr('id') + "']");
            if (label !== undefined && label.attr('for') !== undefined) {
                var r = new RegExp(label.attr('for'), "g");
                label.attr('for', label.attr('for').replace(r, newID).trim());
            }
            thing.prop('id', thing.prop('id').replace(n, n - 1));
        }
    }
}

function replace_app(selector) {
    /*
    Replace the app type.
    Removes the original app from database.
     */
    var n = get_n(selector);
    var type = $("#" + selector).val();
    if (confirm('Replace app?')) {
        $.ajax({
            url: '/general/ajax/replace_app/', data: {
                'n': n, 'type': type,
            }, dataType: 'json', success: function (data) {
                close_sidebar('app-sidebar-' + n);
                $("#" + data.appDistrict).replaceWith(data.newApp);
                protect('sidebar-' + n);
                sidebar_click('sidebar-' + n);
                protect('app-sidebar-' + n);
                sidebar_click('app-sidebar-' + n);
            }
        });
    } else {
        $("#" + selector).val($("#" + selector).data('val'));
    }
}

function replace_occurrence(string, regex, n, replace) {
    var i = 0;
    return string.replace(regex, function (match) {
        i += 1;
        if (i === n) return replace;
        return match;
    });
}

function ts_create_object(node) {
    let extraData = {};
    let success = function (data) {
        node.prototype.metadata['tsObjectID'] = data.tsObjectID;
        node.prototype.setOutputData(0, data.tsObjectID);
    };
    jax_call(node, urls['create'], extraData, success);
}

function ts_create_object(node, url) {
    let data = {
        'dashboardID': $('#dashboardID').text(),
        'objectType': node.title,
    };
    let success = function (data) {
        node.id = data.id;
        node.prototype.setOutputData(0, node.id);
    };
    jax_call(node, url, data, success);
}

function ts_delete_object(node, url) {
    let data = {
        'dashboardID': $('#dashboardID').text(),
        'objectType': node.title,
        'objectID': node.id,
    };
    let success = function (data) {
    };
    jax_call(node, url, data, success);
}

function ts_delete_object(node) {
    let extraData = {};
    let success = function (data) {
        ts_reset_sidebars();
    };
    jax_call(node, urls['delete'], extraData, success);
}

function ts_execute_object(node, url) {
    $.ajax({
        async: false, url: url, data: {
            'dashboardID': $('#dashboardID').text(),
            'objectType': node.title,
            'objectID': node.id,
        }, dataType: 'json', success: function (data) {
        }
    });
}

function ts_reset_sidebars() {
    close_sidebars();
    $(tsSidebarHeadingLeftID).html('');
    $(tsSidebarHeadingRightID).html('');
    Object.keys(sidebarsObj).forEach(function (key) {
        $(sidebarsObj[key]).html('');
    });
}

function ts_reset_sidebars() {
    $(tsSidebarLeftID).html('');
    $(tsSidebarRightID).html('');
}

function ts_set_selected_node(node) {
    _focused_node = node;
}

function ts_update_app_type() {
    var x = 1;
}

function ts_update_focused_node(fieldID, property) {
    let value = $('#' + fieldID).val();
    _focused_node.prototype.setProperty(property, value);
}

function ts_update_input(node, value) {
    let extraData = {
        'value': value,
    };
    let success = function (data) {
        if (node === _focused_node) {
            ts_update_sidebars(node);
        }
    };
    jax_call(node, urls['updateInput'], extraData, success);
}

function ts_update_input(node, url, value) {
    let data = {
        'dashboardID': $('#dashboardID').text(),
        'objectType': node.title,
        'objectID': node.id,
        'value': value,
    };
    let success = function (data) {
    };
    jax_call(node, url, data, success);
}

function ts_update_object(node, url, property, value) {
    let data = {
        'dashboardID': $('#dashboardID').text(),
        'objectType': node.title,
        'objectID': node.id,
        'property': property,
        'value': value,
    };
    let success = function (data) {
    };
    jax_call(node, url, data, success);
}

function ts_update_object(node, property, value) {
    let extraData = {
        'property': property,
        'value': value,
    };
    let success = function (data) {
        if (node === _focused_node) {
            ts_update_sidebars(node);
        }
    };
    jax_call(node, urls['updateProperty'], extraData, success);
}

function ts_update_sidebars(node) {
    let extraData = {};
    let success = function (data) {
        $(tsSidebarHeadingLeftID).html(data['headingLeft']);
        $(tsSidebarHeadingRightID).html(data['headingRight']);
        Object.keys(sidebarsObj).forEach(function (key) {
            $(sidebarsObj[key]).html(data[key])
        });
        open_sidebars();
    };
    jax_call(node, urls['updateSidebars'], extraData, success);
}

$('#app_body').on('change', function () {
    // $("select[id*='model-algorithms'][multiple]:not('[id*=settings]') > option").mousedown(function (e) {
    JIntegral();
});

$('.dropdown-menu a.dropdown-toggle').on('click', function (e) {
    if (!$(this).next().hasClass('show')) {
        $(this).parents('.dropdown-menu').first().find('.show').removeClass("show");
    }
    var $subMenu = $(this).next(".dropdown-menu");
    $subMenu.toggleClass('show');
    $(this).parents('li.nav-item.dropdown.show').on('hidden.bs.dropdown', function (e) {
        $('.dropdown-submenu .show').removeClass("show");
    });
    return false;
});

$('#insert-app-modal').on('shown.bs.modal', function () {
    $('#insert-app-modal').trigger('focus')
});
