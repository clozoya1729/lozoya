function upload_file() {
    var files = document.getElementById('upload-container').files;
    var formData = new FormData();
    var filenames = [];
    for (var i = 0; i < files.length; i++) {
        formData.append(files[i].name, files[i]);
        filenames.push(files[i].name);
    }
    formData.append('csrfmiddlewaretoken', $("[name=csrfmiddlewaretoken]").val());
    formData.append('filenames', filenames);
    $.ajax({
        async: false,
        type: 'POST',
        url: '/general/ajax/upload_file/',
        cache: false,
        processData: false,
        contentType: false,
        data: formData,
        dataType: 'json',
        success: function (data) {
            if (data.msg === undefined) {
                $('#files-list').html(data.filesList);
            } else {
                alert(data.msg);
            }
        }
    });
}

function file_upload_dialog() {
    document.getElementById('upload-container').click();
}

function delete_file(filename) {
    if (confirm("Delete file " + filename + "?")) {
        $.ajax({
            async: false,
            url: '/general/ajax/delete_file/',
            data: {
                'filename': filename
            },
            dataType: 'json',
            success: function (data) {
                $('#files-list').html(data.filesList);
            }
        });
    }
}

function download_button(m) {
    /*
    Shows the download button if a model csv exists.
     */
    if (m === 'show') {
        document.getElementById("download_model").disabled = false;
        $("#download_model").css('display', 'inline');
    }
    if (m === 'hide') {
        document.getElementById("download_model").disabled = true;
        $("#download_model").css('display', 'none');
    }
}

$("#download_model").click(function () {
    $.ajax({
        url: '/ml/ajax/download_model/',
        data: {},
        dataType: 'json',
        success: function (data) {
            download_button('hide');
        }
    });
});

function edit_file(filename) {
    alert('Under construction!');
}

document.getElementById('upload-button').addEventListener('click', file_upload_dialog);
