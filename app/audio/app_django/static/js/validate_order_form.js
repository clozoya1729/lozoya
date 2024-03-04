function validate_email() {
    var email = document.getElementById("id_email").value;
    var restricted = "/\!+=-;:'~`";
    if (check_restricted(email, restricted)) {
        clear_email();
        return;
    }
    end = email.slice(email.length - 4, email.length);
    if (end !== '.com' && end !== '.net' && end !== '.org') {
        clear_email();
        return;
    }
    if (email.replace(/[^@]/g, "").length!==1) {
        clear_email();
        return;
    }
    atIndex = email.indexOf('@')
    if (atIndex<1) {
        clear_email();
        return;
    }
    dotIndex = email.indexOf('.')
    if (dotIndex-atIndex==1) {
        clear_email();

    }
}

function check_restricted(src, restricted) {
    return src.split("").some(ch => restricted.indexOf(ch) !== -1);
}

function clear_email() {
    document.getElementById("id_email").value = '';
    alert('Enter a valid email.');
}

function validate_files() {
    var fileList = document.getElementById("id_files");
    for (var i = 0; i < fileList.files.length; ++i) {
        var name = fileList.files.item(i).name;
        if (fileList.files.length > 128) {
            alert("You may only upload up to 128 files per project.");
            clear_files()
        }
        if (validate_extension(name) === false) {
            alert(name + " is not a valid audio file. Try again.");
            clear_files()
        }
        if (validate_number_of_extensions(name) === false) {
            alert(name + " has multiple extensions.\nMake sure there is only one extension in the file name.");
            clear_files()
        }
    }
}

function clear_files() {
    document.getElementById('id_files').value = '';
}

function validate_extension(name) {
    var extension = name.slice(name.length - 4, name.length);
    if (extension == '.wav' || extension == '.mp3') {
        return true;
    }
    return false;
}

function validate_number_of_extensions(name) {
    var dots = name.split('.').length - 1;
    return (dots === 1);
}

function get_size() {
    var conversions = {
        '.wav': 176.4,
        '.mp3': 24.65,
    };
    var name;
    var extension;
    var sizeKB;
    var duration;
    var longestDuration = 0;
    var fileList = document.getElementById("id_files");
    for (var i = 0; i < fileList.files.length; ++i) {
        name = fileList.files.item(i).name;
        extension = name.slice(name.length - 4, name.length);
        sizeKB = fileList.files.item(i).size / 1000;
        duration = (sizeKB / conversions[extension]) / 60;
        if (duration > longestDuration) {
            longestDuration = duration;
        }
    }
    return longestDuration;
}

function do_math(n, L, t) {
    var pN = 1;
    var pL = 1;
    if (n > 30) {
        pN = 1.5;
    }
    if (L > 30) {
        pL = 1.5;
    }
    var a = 39.57 * Math.log(n + 1);
    var b = 48 * (1.9 - Math.abs(Math.log(L)));
    var c = (t / 21) * Math.log(t);
    var result = pN * pL * (75 + a + b - c).toFixed(2);
    if (result <= 0) {
        result = 0;
    }
    return result;
}

function time_difference() {
    var today = new Date();
    var date_due = document.getElementById("id_date_due").value;
    if (date_due == "") {
        document.getElementById("id_date_due").value = document.getElementById("id_date_due").placeholder;
        date_due = document.getElementById("id_date_due").placeholder;
    }
    var date_due = new Date(date_due);
    var timeDiff = Math.abs(date_due.getTime() - today.getTime());
    var days = Math.ceil(timeDiff / (1000 * 3600 * 24));
    return days;
}

function calculate_price() {
    var numFiles = document.getElementById("id_files").files.length;
    var minutes = get_size();
    var time = time_difference();
    var price = do_math(numFiles, minutes, time);
    document.getElementById("id_price").value = "$" + price;
}


