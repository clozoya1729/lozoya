function get_file_list() {
    return $('#id_files').prop('files');
}

function set_duration(duration) {
    $('#id_duration').val(duration)
}

function reset_longest_duration() {
    $('#longest_duration').text(0);
}

function calculate_price() {
    reset_longest_duration();
    var fileList = get_file_list();
    var numFiles = fileList.length;
    reset_original_price();
    update_price_bar();
    var time = time_difference();
    for (var i = 0; i < numFiles; ++i) {
        get_(fileList.item(i)).then((duration) => {
            var price = do_math(numFiles, duration, time);
            var current = remove_currency(get_original_price());
            if (price > current) {
                p = (price * 4).toFixed(2);
                set_original_price('$' + p);
                set_duration(duration.toFixed(2));
                update_price_bar();
            }
        });
    }
}


function get_(file) {
    let node = document.createElement('audio');
    let promise = new Promise(function (resolve, reject) {
        node.addEventListener('loadedmetadata', function () {
            resolve(node.duration);
        });
        node.addEventListener('error', function () {
            reject(node.error.message + '(' + node.error.code + ')');
        });
    });
    const URL = window.URL || window.webkitURL;
    node.src = URL.createObjectURL(file);
    return promise;
}


function do_math(n, L, t) {
    var pN = 1;
    var pL = 1;
    if (n > 30) {
        pN = 1.5;
    }
    if (L > 600) {
        pL = 1.5;
    }
    var a = 39.57 * Math.log(n + 1);
    var b = 10 + 0.6 * L;
    var c = (t / 21) * Math.log(t);
    var result = pN * pL * (75 + a + b - c).toFixed(2);
    if (result <= 0) {
        result = 0;
    }
    return result;
}

function get_date_due() {
    if ($('#id_date_due').val() === '') {
        reset_date_due();
    }
    return $('#id_date_due').val();
}

function get_date_today() {
    return new Date();
}

function reset_date_due() {
    set_date_due(get_date_due_placeholder());
}

function set_date_due(date) {
    $('#id_date_due').val(date);
}

function get_date_due_placeholder() {
    return $('#id_date_due').attr('placeholder');
}

function time_difference() {
    var today = get_date_today();
    var date_due = new Date(get_date_due());
    var timeDiff = Math.abs(date_due.getTime() - today.getTime());
    var days = Math.ceil(timeDiff / (86400000)); //1000*3600*24
    return days;
}

function get_file_list() {
    return $('#id_files').prop('files');
}

function set_duration(duration) {
    $('#id_duration').val(duration)
}

function reset_longest_duration() {
    $('#longest_duration').text(0);
}

function calculate_price() {
    reset_longest_duration();
    var fileList = get_file_list();
    var numFiles = fileList.length;
    reset_original_price();
    update_price_bar();
    var time = time_difference();
    for (var i = 0; i < numFiles; ++i) {
        get_(fileList.item(i)).then((duration) => {
            var price = do_math(numFiles, duration, time);
            var current = remove_currency(get_original_price());
            if (price > current) {
                p = (price * 4).toFixed(2);
                set_original_price('$' + p);
                set_duration(duration.toFixed(2));
                update_price_bar();
            }
        });
    }
}


function get_(file) {
    let node = document.createElement('audio');
    let promise = new Promise(function (resolve, reject) {
        node.addEventListener('loadedmetadata', function () {
            resolve(node.duration);
        });
        node.addEventListener('error', function () {
            reject(node.error.message + '(' + node.error.code + ')');
        });
    });
    const URL = window.URL || window.webkitURL;
    node.src = URL.createObjectURL(file);
    return promise;
}


function do_math(n, L, t) {
    var pN = 1;
    var pL = 1;
    if (n > 30) {
        pN = 1.5;
    }
    if (L > 600) {
        pL = 1.5;
    }
    var a = 39.57 * Math.log(n + 1);
    var b = 10 + 0.6 * L;
    var c = (t / 21) * Math.log(t);
    var result = pN * pL * (75 + a + b - c).toFixed(2);
    if (result <= 0) {
        result = 0;
    }
    return result;
}

function get_date_due() {
    if ($('#id_date_due').val() === '') {
        reset_date_due();
    }
    return $('#id_date_due').val();
}

function get_date_today() {
    return new Date();
}

function reset_date_due() {
    set_date_due(get_date_due_placeholder());
}

function set_date_due(date) {
    $('#id_date_due').val(date);
}

function get_date_due_placeholder() {
    return $('#id_date_due').attr('placeholder');
}

function time_difference() {
    var today = get_date_today();
    var date_due = new Date(get_date_due());
    var timeDiff = Math.abs(date_due.getTime() - today.getTime());
    var days = Math.ceil(timeDiff / (86400000)); //1000*3600*24
    return days;
}

