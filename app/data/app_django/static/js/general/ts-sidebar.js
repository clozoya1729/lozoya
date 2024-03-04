$(document).ready(function () {
    $("#sidebar").niceScroll({
        cursorcolor: 'rgba(0,0,0,.3)',
        cursorwidth: 0,
        cursorborder: 'none'
    });
    $('#sidebar a').on('click', function () {
        setTimeout(function () {
            $("#sidebar").getNiceScroll().resize();
        }, 500);
    });
    $('#close-sidebar, .overlay').on('click', function () {
        $('#sidebar').removeClass('active');
        $('.overlay').fadeOut();
    });
    $('.open-sidebar').on('click', function () {
        $('#sidebar').addClass('active');
        $('.overlay').fadeIn();
        // $('.collapse.in').toggleClass('in');
        // $('a[aria-expanded=true]').attr('aria-expanded', 'false');
        $('#sidebar').attr('aria-expanded', 'false');
        $('#sidebar').toggleClass('in');
    });
});

function change_sidebar_menu(urlRoot, appType, menu, n) {
    $.ajax({
        url: '/' + urlRoot + '/' + appType + '/ajax/change_sidebar_menu/', data: {
            'appType': appType, 'menu': menu, 'n': n,
        }, dataType: 'json', success: function (data) {
            $("#sidebar_body").html(data.body);
        }
    });
}

function close_sidebar(field) {
    $('#' + field).removeClass('active');
}

function close_sidebars() {
    $('#ts-sidebar-left').removeClass('active');
    $('#ts-sidebar-right').removeClass('active');
}

function open_sidebar(field) {
    $('#' + field).addClass('active');
    $('a[aria-expanded=true]').attr('aria-expanded', 'false');
}

function open_sidebars() {
    $('#ts-sidebar-left').addClass('active');
    $('#ts-sidebar-right').addClass('active');
    $('.collapse.in').toggleClass('in');
    $('a[aria-expanded=true]').attr('aria-expanded', 'false');
}

function populate_sidebar(urlRoot, appType, n) {
    $.ajax({
        url: '/' + urlRoot + '/' + appType + '/ajax/populate_sidebar/', data: {
            'appType': appType, 'n': n,
        }, dataType: 'json', success: function (data) {
            $("#sidebar_navbar").html(data.navbar);
            $("#sidebar_header").html(data.header);
            $("#sidebar_body").html(data.body);
            JIntegral();
        }
    });
}

function protect(field) {
    $("#" + field).niceScroll({
        cursorcolor: 'rgba(0,0,0,.3)', cursorwidth: 0, cursorborder: 'none',
    });
}

function sidebar_click(field) {
    $('#' + field + ' a').on('click', function () {
        setTimeout(function () {
            $("#" + field).getNiceScroll().resize();
        }, 500);
    });
}

function set_sidebar_scroll() {
    $("#ts-sidebar-left").niceScroll({
        cursorcolor: 'rgba(0,0,0,.3)', cursorwidth: 0, cursorborder: 'none',
    });
    $("#ts-sidebar-right").niceScroll({
        cursorcolor: 'rgba(0,0,0,.3)', cursorwidth: 0, cursorborder: 'none',
    });
}

function set_sidebar_timeout() {
    $('#ts-sidebar-left a').on('click', function () {
        setTimeout(function () {
            $("#ts-sidebar-left").getNiceScroll().resize();
        }, 500);
    });
    $('#ts-sidebar-right a').on('click', function () {
        setTimeout(function () {
            $("#ts-sidebar-right").getNiceScroll().resize();
        }, 500);
    });
}

function update_sidebar_navbar(groupID, clicked) {
    $("[id*=" + groupID + "]").removeClass('active-tab');
    $('#' + clicked).addClass('active-tab');
}

$(document).ready(function () {
    set_sidebar_scroll();
    set_sidebar_timeout();
});

