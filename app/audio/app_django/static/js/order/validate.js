function validate_email() {
    var email = $("#id_email").val();
    var restricted = "/;:()";
    if (check_restricted(email, restricted)) {
        reset_email();
    }
    var end = email.slice(email.length - 4, email.length);
    if (end !== '.edu' && end !== '.com' && end !== '.net' && end !== '.org') {
        reset_email();
    }
    if (email.replace(/[^@]/g, "").length !== 1) {
        reset_email();
    }
    var atIndex = email.indexOf('@')
    if (atIndex < 1) {
        reset_email();
    }
    dotIndex = email.indexOf('.')
    if (dotIndex - atIndex == 1) {
        reset_email();
    }
}

function check_restricted(src, restricted) {
    return src.split("").some(ch => restricted.indexOf(ch) !== -1);
}

function reset_email() {
    $('#id_email').val('');
    alert('Enter a valid email.');
}

function validate_files() {
    var fileList = get_file_list();
    for (var i = 0; i < fileList.length; ++i) {
        var name = fileList.item(i).name;
        if (fileList.length > 128) {
            alert("You may only upload up to 128 files per project.");
            reset_files()
        }
        if (validate_extension(name) === false) {
            alert(name + " is not a valid audio file. Try again.");
            reset_files()
        }
        if (validate_number_of_extensions(name) === false) {
            alert(name + " has multiple extensions.\nMake sure there is only one extension in the file name.");
            reset_files()
        }
    }
}

function reset_files() {
    $('#id_files').val('');
}

function validate_extension(name) {
    var extension = name.slice(name.length - 4, name.length);
    return (extension == '.wav' || extension == '.mp3');
}

function validate_number_of_extensions(name) {
    var dots = name.split('.').length - 1;
    return (dots === 1);
}

function validate_email() {
    var email = $("#id_email").val();
    var restricted = "/;:()";
    if (check_restricted(email, restricted)) {
        reset_email();
    }
    var end = email.slice(email.length - 4, email.length);
    if (end !== '.edu' && end !== '.com' && end !== '.net' && end !== '.org') {
        reset_email();
    }
    if (email.replace(/[^@]/g, "").length !== 1) {
        reset_email();
    }
    var atIndex = email.indexOf('@')
    if (atIndex < 1) {
        reset_email();
    }
    dotIndex = email.indexOf('.')
    if (dotIndex - atIndex == 1) {
        reset_email();
    }
}

function check_restricted(src, restricted) {
    return src.split("").some(ch => restricted.indexOf(ch) !== -1);
}

function reset_email() {
    $('#id_email').val('');
    alert('Enter a valid email.');
}

function validate_files() {
    var fileList = get_file_list();
    for (var i = 0; i < fileList.length; ++i) {
        var name = fileList.item(i).name;
        if (fileList.length > 128) {
            alert("You may only upload up to 128 files per project.");
            reset_files()
        }
        if (validate_extension(name) === false) {
            alert(name + " is not a valid audio file. Try again.");
            reset_files()
        }
        if (validate_number_of_extensions(name) === false) {
            alert(name + " has multiple extensions.\nMake sure there is only one extension in the file name.");
            reset_files()
        }
    }
}

function reset_files() {
    $('#id_files').val('');
}

function validate_extension(name) {
    var extension = name.slice(name.length - 4, name.length);
    return (extension == '.wav' || extension == '.mp3');
}

function validate_number_of_extensions(name) {
    var dots = name.split('.').length - 1;
    return (dots === 1);
}
