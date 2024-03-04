var colorFontGood = 'green';
var colorFontBad = 'red';
var boxSize = '1px 1px 3px 3px';
var colorShadowGood = boxSize + ' ' + colorFontGood;
var colorShadowBad = boxSize + ' ' + colorFontBad;

function hide_original_price() {
    $('#label_original_price_div').css('display', 'none');
}


function show_original_price() {
    $('#label_original_price_div').css('display', 'inline');
}


function hide_discount() {
    $('#label_discount_div').css('display', 'none');
}


function show_discount() {
    $('#label_discount_div').css('display', 'inline');
}


function set_valid_colors() {
    $('#id_coupon_message').css('color', colorFontGood);
    $('#id_coupon_redeemed').css('boxShadow', colorShadowGood);
}


function set_invalid_colors() {
    $('#id_coupon_message').css('color', colorFontBad);
    $('#id_coupon_redeemed').css('boxShadow', colorShadowBad);
}


function set_blank_colors() {
    $('#id_coupon_message').css('color', 'transparent');
    $('#id_coupon_redeemed').css('boxShadow', 'none');
}

var colorFontGood = 'green';
var colorFontBad = 'red';
var boxSize = '1px 1px 3px 3px';
var colorShadowGood = boxSize + ' ' + colorFontGood;
var colorShadowBad = boxSize + ' ' + colorFontBad;

function hide_original_price() {
    $('#label_original_price_div').css('display', 'none');
}


function show_original_price() {
    $('#label_original_price_div').css('display', 'inline');
}


function hide_discount() {
    $('#label_discount_div').css('display', 'none');
}


function show_discount() {
    $('#label_discount_div').css('display', 'inline');
}


function set_valid_colors() {
    $('#id_coupon_message').css('color', colorFontGood);
    $('#id_coupon_redeemed').css('boxShadow', colorShadowGood);
}


function set_invalid_colors() {
    $('#id_coupon_message').css('color', colorFontBad);
    $('#id_coupon_redeemed').css('boxShadow', colorShadowBad);
}


function set_blank_colors() {
    $('#id_coupon_message').css('color', 'transparent');
    $('#id_coupon_redeemed').css('boxShadow', 'none');
}