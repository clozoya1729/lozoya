function get_form() {
    return $('#id_order_form');
}


function get_url() {
    return get_form().attr('data-validate-coupon-url');
}


function get_coupon() {
    return $('#id_coupon_redeemed').val();
}


function _get_price() {
    return $('#id_price').val();
}


function get_original_price() {
    if (_get_price() === '') {
        reset_original_price();
    }
    return _get_price();
}


function get_final_price() {
    return $('#label_final_price').val();
}


function remove_currency(price) {
    return parseFloat(price.slice(1, price.length));
}


function remove_percent(percentage) {
    return parseFloat(percentage.slice(0, percentage.length - 1))
}

function _get_discount() {
    return $('#id_discount_redeemed').val();
}


function get_discount() {
    discount = _get_discount();
    if (discount.indexOf('%') > -1 && remove_percent(discount) > 100) {
        set_discount('100%');
    } else if (discount === '') {
        reset_discount();
    }
    return _get_discount();
}


function get_discount_value() {
    var discount = get_discount();
    var priceValue = remove_currency(get_original_price());
    if (discount.indexOf('$') > -1) {
        return remove_currency(discount);
    } else if (discount.indexOf('%') > -1) {
        return priceValue * remove_percent(discount) / 100;
    } else {
        reset_discount();
        set_final_price(remove_currency(get_original_price()), 0.00);
        return 0.00;
    }
}

function get_coupon_message() {
    return $('#id_coupon_message').val();
}

function get_form() {
    return $('#id_order_form');
}


function get_url() {
    return get_form().attr('data-validate-coupon-url');
}


function get_coupon() {
    return $('#id_coupon_redeemed').val();
}


function _get_price() {
    return $('#id_price').val();
}


function get_original_price() {
    if (_get_price() === '') {
        reset_original_price();
    }
    return _get_price();
}


function get_final_price() {
    return $('#label_final_price').val();
}


function remove_currency(price) {
    return parseFloat(price.slice(1, price.length));
}


function remove_percent(percentage) {
    return parseFloat(percentage.slice(0, percentage.length - 1))
}

function _get_discount() {
    return $('#id_discount_redeemed').val();
}


function get_discount() {
    discount = _get_discount();
    if (discount.indexOf('%') > -1 && remove_percent(discount) > 100) {
        set_discount('100%');
    } else if (discount === '') {
        reset_discount();
    }
    return _get_discount();
}


function get_discount_value() {
    var discount = get_discount();
    var priceValue = remove_currency(get_original_price());
    if (discount.indexOf('$') > -1) {
        return remove_currency(discount);
    } else if (discount.indexOf('%') > -1) {
        return priceValue * remove_percent(discount) / 100;
    } else {
        reset_discount();
        set_final_price(remove_currency(get_original_price()), 0.00);
        return 0.00;
    }
}

function get_coupon_message() {
    return $('#id_coupon_message').val();
}
