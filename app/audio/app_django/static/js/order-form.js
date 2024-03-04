$(function() {
    $( ".datepicker" ).datepicker({
	//hideIfNoPrevNext: true,
	//showOn: 'button',
	//showOtherMonths: true,
	//minDate: "+7D",
	//maxDate: "+6M"
    });
});


function validate_files(){
    var fileList = document.getElementById("id_file");
    for (var i=0; i< fileList.files.length; ++i) {
	var name = fileList.files.item(i).name;
	if (validate_extension(name) === false) {
	  alert(name+" is not a valid audio file. Try again.");
	  document.getElementById('id_file').value = '';
	}
	if (validate_number_of_extensions(name) === false) {
	  alert(name+" has multiple extensions. Make sure there is only one extension in the file name.");
	  document.getElementById('id_file').value = '';
	}
    }
}


function validate_extension(name) {
    var extension = name.slice(name.length-4, name.length);
    if (extension=='.wav'||extension=='.mp3') {
	return true;
    }
    return false;
  }


function validate_number_of_extensions(name) {
    var dots = name.split('.').length - 1;
    if (dots===1) {
	return true;
    }
    return false;
}


function calculate_price() {
    numFiles = document.getElementById("id_file").files.length
    document.getElementById("id_price").value = "$"+(numFiles*7).toFixed(2);
}


function validate_order_form() {
    var lisht = ["id_artist_name",
		 "id_song_title",
		 "id_details",
		 "id_reference_artist",
		 "id_reference_song"];
    lisht.forEach(function(element) {
      if (document.getElementById(element).value=="") {
	document.getElementById(element).value = "";
      }
    });
}


function validate_date_due() {
    alert(document.getItemById("id_date-due").value);
    if (document.getItemById("id_date_due").value=="") {
	var now = new Date();
	alert(now);
	var nextWeek = new Date(now);
	alert(nextWeek);
	nextWeek.setDate(nextWeek.getDate()+7);
	alert(nextWeek);
	document.getElementById("id_date-due").value = nextWeek;
    }
}
