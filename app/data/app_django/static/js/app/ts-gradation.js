//Build Tabulator
var table = new Tabulator("#test-table", {
    height: "311px",
    addRowPos: "bottom",
    columns: [
        {title: "Name", field: "name", width: 200, editor: "input"},
        {title: "Progress", field: "progress", width: 100, hozAlign: "right", sorter: "number", editor: "input"},
        {title: "Gender", field: "gender", editor: "input"},
        {title: "Rating", field: "rating", hozAlign: "center", width: 80, editor: "input"},
        {title: "Favourite Color", field: "col", editor: "input"},
        {title: "Date Of Birth", field: "dob", hozAlign: "center", sorter: "date", editor: "input"},
        {title: "Driver", field: "car", hozAlign: "center", editor: "input"},
    ],
});

//Add row on "Add Row" button click
document.getElementById("add-row").addEventListener("click", function () {
    table.addRow({});
});

//Delete row on "Delete Row" button click
document.getElementById("del-row").addEventListener("click", function () {
    table.deleteRow(1);
});

//Clear table on "Empty the table" button click
document.getElementById("clear").addEventListener("click", function () {
    table.clearData()
});

//Reset table contents on "Reset the table" button click
document.getElementById("reset").addEventListener("click", function () {
    table.setData(tabledata);
});