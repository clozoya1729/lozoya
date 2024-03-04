$('#server-configuration-slider').mousemove(function () {
    var cpu = [1, 1, 2, 4, 6, 8, 12, 16, 20, 24, 32];
    var price = [15, 20, 30, 50, 90, 170, 250, 330, 490, 650, 970];
    var ram = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 196];
    var storage = [25, 50, 80, 160, 320, 640, 960, 1280, 1920, 2560, 3840];
    var transfer = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    $('#server-configuration-label').html(
        cpu[this.value] + "<br>"
        + ram[this.value] + " GB<br>"
        + storage[this.value] + " GB<br>"
        + transfer[this.value] + " TB<br>"
    );
    $('#price-label').html(
        "$" + price[this.value] + " Monthly"
    );
});
