$('#id_coupon_redeemed').change(function () {
    update_price_bar();
});


function update_price_bar() {
    var coupon = get_coupon();
    var url = get_url();
    if (coupon !== '') {
        $.ajax({
            'url': url, 'data': {'coupon': coupon}, 'dataType': 'json', 'success': function (data) {
                if (data.isValid) {
                    valid_coupon(data.discount);
                } else {
                    invalid_coupon();
                }
            },
        });
    } else {
        blank_coupon();
    }
}


function blank_coupon() {
    set_label_discount('');
    set_coupon_message('');
    set_blank_colors();
    reset_discount();
    reset_final_price();
    hide_original_price();
    hide_discount();
}


function invalid_coupon() {
    set_label_discount('');
    set_coupon_message('Invalid Coupon');
    set_invalid_colors();
    set_original_price(get_original_price());
    reset_discount();
    reset_final_price();
    hide_original_price();
    hide_discount();
}


function valid_coupon(discount) {
    set_label_discount('-' + discount + '=');
    set_coupon_message(discount + ' Off!');
    set_valid_colors();
    set_original_price(get_original_price());
    set_discount(discount);
    set_final_price(remove_currency(get_original_price()), get_discount_value());
    show_original_price();
    show_discount();
}

$('#id_coupon_redeemed').change(function () {
    update_price_bar();
});


function update_price_bar() {
    var coupon = get_coupon();
    var url = get_url();
    if (coupon !== '') {
        $.ajax({
            'url': url, 'data': {'coupon': coupon}, 'dataType': 'json', 'success': function (data) {
                if (data.isValid) {
                    valid_coupon(data.discount);
                } else {
                    invalid_coupon();
                }
            },
        });
    } else {
        blank_coupon();
    }
}


function blank_coupon() {
    set_label_discount('');
    set_coupon_message('');
    set_blank_colors();
    reset_discount();
    reset_final_price();
    hide_original_price();
    hide_discount();
}


function invalid_coupon() {
    set_label_discount('');
    set_coupon_message('Invalid Coupon');
    set_invalid_colors();
    set_original_price(get_original_price());
    reset_discount();
    reset_final_price();
    hide_original_price();
    hide_discount();
}


function valid_coupon(discount) {
    set_label_discount('-' + discount + '=');
    set_coupon_message(discount + ' Off!');
    set_valid_colors();
    set_original_price(get_original_price());
    set_discount(discount);
    set_final_price(remove_currency(get_original_price()), get_discount_value());
    show_original_price();
    show_discount();
}
