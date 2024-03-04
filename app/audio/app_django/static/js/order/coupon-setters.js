function set_original_price(price) {
    $('#label_original_price').val(price);
    $('#id_price').val(price);
}


function set_discount(discount) {
    $('#id_discount_redeemed').val(discount);
}


function set_label_discount(discount) {
    $('#label_discount').val(discount);
}


function set_final_price(price, discount) {
    if (discount < price) {
        $('#label_final_price').val('$' + (price - discount));
    } else {
        set_discount('$' + price);
        $('#label_final_price').val('$0.00');
    }
}


function set_coupon_message(message) {
    $('#id_coupon_message').text(message);
}


function reset_original_price() {
    set_original_price('$0.00');
}


function reset_discount() {
    set_discount('$0.00');
}


function reset_final_price() {
    set_final_price(remove_currency(get_original_price()), 0.00);
}

function set_original_price(price) {
    $('#label_original_price').val(price);
    $('#id_price').val(price);
}


function set_discount(discount) {
    $('#id_discount_redeemed').val(discount);
}


function set_label_discount(discount) {
    $('#label_discount').val(discount);
}


function set_final_price(price, discount) {
    if (discount < price) {
        $('#label_final_price').val('$' + (price - discount));
    } else {
        set_discount('$' + price);
        $('#label_final_price').val('$0.00');
    }
}


function set_coupon_message(message) {
    $('#id_coupon_message').text(message);
}


function reset_original_price() {
    set_original_price('$0.00');
}


function reset_discount() {
    set_discount('$0.00');
}


function reset_final_price() {
    set_final_price(remove_currency(get_original_price()), 0.00);
}
