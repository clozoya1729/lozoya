window.onload = function () {
    var counter = 0, canvas = document.getElementById("canvas"), canvas2 = document.getElementById("canvas2"),
        canvas3 = document.getElementById("canvas3"), canvas4 = document.getElementById("canvas4"),
        context = canvas.getContext("2d"), context2 = canvas2.getContext("2d"), context3 = canvas3.getContext("2d"),
        context4 = canvas4.getContext("2d"),
        width = canvas.width = canvas2.width = canvas3.width = canvas4.width = window.innerWidth,
        height = canvas.height = canvas2.height = canvas3.height = canvas4.height = window.innerHeight, drawing = true;


    var arm = Arm.create(width / 2, height / 2, 100, 0), angle = 0, angle2 = 0, angle3 = 0.605,
        arm2 = Arm.create(arm.getEndX(), arm.getEndY(), 200, 1.3),
        arm3 = Arm.create(arm2.getEndX(), arm2.getEndY(), 100, 1.3), arm4 = Arm.create(width / 4, height / 4, 100, -1)
    arm5 = Arm.create(arm4.getEndX(), arm4.getEndY(), 100, 0.3), arm6 = Arm.create(arm5.getEndX(), arm5.getEndY(), 200, 0.3), arm7 = Arm.create(2 * width / 2.5, height / 2.1, 100, -1)
    arm8 = Arm.create(arm7.getEndX(), arm7.getEndY(), 100, 0.3), arm9 = Arm.create(arm8.getEndX(), arm8.getEndY(), 200, 0.3), arm10 = Arm.create(arm9.getEndX(), arm9.getEndY(), 100, 0.3),

        arm2.parent = arm;
    arm3.parent = arm2;
    arm4.parent = arm3;
    arm5.parent = arm4;
    arm6.parent = arm5;
    arm7.parent = arm6;
    arm8.parent = arm7;
    arm9.parent = arm8;
    arm10.parent = arm9;
    context2.lineWidth = 0.05;
    context3.lineWidth = 0.05;
    context4.lineWidth = 0.05;

    update();
    document.body.addEventListener("click", function () {
        drawing = true;
    })

    function update() {
        if (drawing) {
            counter += 0.00000001 + 0.1 * Math.abs(Math.sin(counter / 2));
            context2.beginPath();
            context2.moveTo(arm3.getEndX(), arm3.getEndY());
            context3.beginPath();
            context3.moveTo(arm6.getEndX(), arm6.getEndY());
            context4.beginPath();
            context4.moveTo(arm10.getEndX(), arm10.getEndY());
        }

        context2.lineWidth = 0.10 + 0.60 * Math.abs(Math.sin(counter)) - 0.10 * (Math.sin(0.5 * counter));
        context3.lineWidth = 0.10 + 0.50 * Math.abs(Math.sin(counter)) + 0.20 * (Math.sin(0.5 * counter));
        context4.lineWidth = 0.10 + 0.40 * Math.abs(Math.sin(counter)) - 0.30 * (Math.sin(0.5 * counter));
        context.clearRect(0, 0, width, height);
        arm.angle = Math.sin(angle3) * 1.476;
        arm2.angle = Math.cos(angle2 * .502 + 2) * 2.92;
        arm3.angle = Math.sin(angle * 1.498 - 0.5) * 0.34;
        arm4.angle = Math.sin(angle2) * 2.476;
        arm5.angle = Math.cos(angle3 * .502 + 2) * 2.92;
        arm6.angle = Math.sin(angle * 1.498 - 0.5) * 2.34;
        arm7.angle = 0.5 * Math.sin(angle) * 3.1415 / 4;
        arm8.angle = 0.5 * Math.cos(angle3 * .502 + 2) * 0.92;
        arm9.angle = 0.1 * Math.sin(angle2 * 0.498 - 0.5) * 0.34;
        arm10.angle = 0.1 * Math.sin(angle) * 2.7181 / 4;
        arm2.x = arm.getEndX();
        arm2.y = arm.getEndY();
        arm3.x = arm2.getEndX() + 100 * Math.cos(counter);
        arm3.y = arm2.getEndY() + 100 * Math.sin(counter);
        arm5.x = arm4.getEndX();
        arm5.y = arm4.getEndY();
        arm6.x = arm5.getEndX() + 100 * Math.sin(counter);
        arm6.y = arm5.getEndY() + 100 * (Math.sin(counter));
        arm8.x = arm7.getEndX();
        arm8.y = arm7.getEndY();
        arm9.x = arm8.getEndX();
        arm9.y = arm8.getEndY();
        arm10.x = arm9.getEndX() - 100 * Math.abs(Math.sin(0.5 * counter));
        arm10.y = arm9.getEndY() - 100 * Math.sin(0.5 * counter);
        angle += 0.05 - (0.005 / (1 + Math.abs(Math.sin(counter))));
        angle2 += 0.01 - (0.005 / (1 + Math.abs(Math.sin(counter))));
        angle3 += 0.027181 - (0.005 / (1 + Math.abs(Math.sin(counter))));
        arm.render(context);
        arm2.render(context);
        arm3.render(context);
        arm4.render(context);
        arm5.render(context);
        arm6.render(context);
        arm7.render(context);
        arm8.render(context);
        arm9.render(context);
        arm10.render(context);

        if (drawing) {
            context2.lineTo(arm3.getEndX(), arm3.getEndY());
            context2.stroke();
            context3.lineTo(arm6.getEndX(), arm6.getEndY());
            context3.stroke();
            context4.lineTo(arm10.getEndX(), arm10.getEndY());
            context4.stroke();
        }
        requestAnimationFrame(update);
    }
}
window.onload = function () {
    var canvas = document.getElementById("canvas"), context = canvas.getContext("2d"),
        width = canvas.width = window.innerWidth, height = canvas.height = window.innerHeight;

    for (var i = 0; i < 100; i += 1) {
        context.beginPath();
        context.moveTo(Math.random() * width, Math.random() * height);
        context.lineTo(Math.random() * width, Math.random() * height);
        context.stroke();
    }
};
/**
 * Created by cocodell on 5/2/2017.
 */

var Arm = Arm || {
    x: 0, y: 0, length: 100, angle: 0, parent: null,

    create: function (x, y, length, angle) {
        var obj = Object.create(this);
        obj.init(x, y, length, angle);
        return obj;
    },

    init: function (x, y, length, angle) {
        this.x = x;
        this.y = y;
        this.length = length;
        this.angle = angle;
    },

    getEndX: function () {
        var angle = this.angle;
        parent = this.parent;
        while (parent) {
            angle += parent.angle;
            parent = parent.parent;
        }
        return this.x + Math.cos(angle) * this.length;
    },

    getEndY: function () {
        var angle = this.angle;
        parent = this.parent;
        while (parent) {
            angle += parent.angle;
            parent = parent.parent;
        }
        return this.y + Math.sin(angle) * this.length;
    },

    render: function (context) {
        context.strokeStyle = "#000000";
        context.lineWidth = 5;
        context.beginPath();
        context.moveTo(this.x, this.y);
        context.lineTo(this.getEndX(), this.getEndY());
        context.stroke();
    }

};

window.onload = function () {
    var counter = 0, canvas = document.getElementById("canvas"), canvas2 = document.getElementById("canvas2"),
        canvas3 = document.getElementById("canvas3"), canvas4 = document.getElementById("canvas4"),
        context = canvas.getContext("2d"), context2 = canvas2.getContext("2d"), context3 = canvas3.getContext("2d"),
        context4 = canvas4.getContext("2d"),
        width = canvas.width = canvas2.width = canvas3.width = canvas4.width = window.innerWidth,
        height = canvas.height = canvas2.height = canvas3.height = canvas4.height = window.innerHeight, drawing = true;


    var arm = Arm.create(width / 2, height / 2, 100, 0), angle = 0, angle2 = 0, angle3 = 0.605,
        arm2 = Arm.create(arm.getEndX(), arm.getEndY(), 100, 1.3),
        arm3 = Arm.create(arm2.getEndX(), arm2.getEndY(), 100, 1.3), arm4 = Arm.create(width / 4, height / 4, 100, -1)
    arm5 = Arm.create(arm4.getEndX(), arm4.getEndY(), 100, 0.3), arm6 = Arm.create(arm5.getEndX(), arm5.getEndY(), 100, 0.3), arm7 = Arm.create(2 * width / 2.5, height / 2.1, 100, -1)
    arm8 = Arm.create(arm7.getEndX(), arm7.getEndY(), 100, 0.3), arm9 = Arm.create(arm8.getEndX(), arm8.getEndY(), 100, 0.3), arm10 = Arm.create(arm9.getEndX(), arm9.getEndY(), 100, 0.3),

        arm2.parent = arm;
    arm3.parent = arm2;
    arm4.parent = arm3;
    arm5.parent = arm4;
    arm6.parent = arm5;
    arm7.parent = arm6;
    arm8.parent = arm7;
    arm9.parent = arm8;
    arm10.parent = arm9;
    context2.lineWidth = 0.05;
    context3.lineWidth = 0.05;
    context4.lineWidth = 0.05;

    update();
    document.body.addEventListener("click", function () {
        drawing = true;
    })

    function update() {
        if (drawing) {
            counter += 0.001 + 0.01 * Math.sin(counter);
            context2.beginPath();
            context2.moveTo(arm3.getEndX(), arm3.getEndY());
            context3.beginPath();
            context3.moveTo(arm6.getEndX(), arm6.getEndY());
            context4.beginPath();
            context4.moveTo(arm10.getEndX(), arm10.getEndY());
        }

        context2.lineWidth = 1 * Math.abs(Math.sin(counter));
        context3.lineWidth = 1 * Math.abs(Math.sin(counter));
        context4.lineWidth = 1 * Math.abs(Math.sin(counter));
        context.clearRect(0, 0, width, height);
        arm.angle = Math.sin(angle3) * 1.476;
        arm2.angle = Math.cos(angle2 * .502 + 2) * 2.92;
        arm3.angle = Math.sin(angle * 1.498 - 0.5) * 0.34;
        arm4.angle = Math.sin(angle2) * 2.476;
        arm5.angle = Math.cos(angle3 * .502 + 2) * 2.92;
        arm6.angle = Math.sin(angle * 1.498 - 0.5) * 2.34;
        arm7.angle = Math.sin(angle) * 2.476;
        arm8.angle = Math.cos(angle3 * .502 + 2) * 2.92;
        arm9.angle = Math.sin(angle2 * 1.498 - 0.5) * 2.34;
        arm10.angle = Math.sin(angle3) * 2.7181;
        arm2.x = arm.getEndX();
        arm2.y = arm.getEndY();
        arm3.x = arm2.getEndX();
        arm3.y = arm2.getEndY();
        arm5.x = arm4.getEndX();
        arm5.y = arm4.getEndY();
        arm6.x = arm5.getEndX();
        arm6.y = arm5.getEndY();
        arm8.x = arm7.getEndX();
        arm8.y = arm7.getEndY();
        arm9.x = arm8.getEndX();
        arm9.y = arm8.getEndY();
        arm10.x = arm9.getEndX();
        arm10.y = arm9.getEndY();
        angle += 0.05;
        angle2 += 0.01;
        angle3 += 0.027181;
        /*arm.render(context);
        arm2.render(context);
        arm3.render(context);
        arm4.render(context);
        arm5.render(context);
        arm6.render(context);
        arm7.render(context);
        arm8.render(context);
        arm9.render(context);
        arm10.render(context);*/

        if (drawing) {
            context2.lineTo(arm3.getEndX(), arm3.getEndY());
            context2.stroke();
            context3.lineTo(arm6.getEndX(), arm6.getEndY());
            context3.stroke();
            context4.lineTo(arm10.getEndX(), arm10.getEndY());
            context4.stroke();
        }
        requestAnimationFrame(update);
    }
}
window.onload = function () {
    var counter = 0, canvas = document.getElementById("canvas"), canvas2 = document.getElementById("canvas2"),
        canvas3 = document.getElementById("canvas3"), canvas4 = document.getElementById("canvas4"),
        context = canvas.getContext("2d"), context2 = canvas2.getContext("2d"), context3 = canvas3.getContext("2d"),
        context4 = canvas4.getContext("2d"),
        width = canvas.width = canvas2.width = canvas3.width = canvas4.width = window.innerWidth,
        height = canvas.height = canvas2.height = canvas3.height = canvas4.height = window.innerHeight, drawing = true;


    var arm = Arm.create(width / 2, height / 2, 100, 0), angle = 0, angle2 = 0, angle3 = 0.605,
        arm2 = Arm.create(arm.getEndX(), arm.getEndY(), 100, 1.3),
        arm3 = Arm.create(arm2.getEndX(), arm2.getEndY(), 100, 1.3), arm4 = Arm.create(width / 4, height / 4, 100, -1)
    arm5 = Arm.create(arm4.getEndX(), arm4.getEndY(), 100, 0.3), arm6 = Arm.create(arm5.getEndX(), arm5.getEndY(), 100, 0.3), arm7 = Arm.create(2 * width / 2.5, height / 2.1, 100, -1)
    arm8 = Arm.create(arm7.getEndX(), arm7.getEndY(), 100, 0.3), arm9 = Arm.create(arm8.getEndX(), arm8.getEndY(), 100, 0.3), arm10 = Arm.create(arm9.getEndX(), arm9.getEndY(), 100, 0.3),

        arm2.parent = arm;
    arm3.parent = arm2;
    arm4.parent = arm3;
    arm5.parent = arm4;
    arm6.parent = arm5;
    arm7.parent = arm6;
    arm8.parent = arm7;
    arm9.parent = arm8;
    arm10.parent = arm9;
    context2.lineWidth = 0.05;
    context3.lineWidth = 0.05;
    context4.lineWidth = 0.05;

    update();
    document.body.addEventListener("click", function () {
        drawing = true;
    })

    function update() {
        if (drawing) {
            counter += 0.00000001 + 0.1 * Math.abs(Math.sin(counter));
            context2.beginPath();
            context2.moveTo(arm3.getEndX(), arm3.getEndY());
            context3.beginPath();
            context3.moveTo(arm6.getEndX(), arm6.getEndY());
            context4.beginPath();
            context4.moveTo(arm10.getEndX(), arm10.getEndY());
        }

        context2.lineWidth = 0.10 + 0.80 * Math.abs(Math.sin(counter));
        context3.lineWidth = 0.10 + 0.80 * Math.abs(Math.sin(counter));
        context4.lineWidth = 0.10 + 0.80 * Math.abs(Math.sin(counter));
        context.clearRect(0, 0, width, height);
        arm.angle = Math.sin(angle3) * 1.476;
        arm2.angle = Math.cos(angle2 * .502 + 2) * 2.92;
        arm3.angle = Math.sin(angle * 1.498 - 0.5) * 0.34;
        arm4.angle = Math.sin(angle2) * 2.476;
        arm5.angle = Math.cos(angle3 * .502 + 2) * 2.92;
        arm6.angle = Math.sin(angle * 1.498 - 0.5) * 2.34;
        arm7.angle = Math.sin(angle) * 3.1415 / 4;
        arm8.angle = Math.cos(angle3 * .502 + 2) * 0.92;
        arm9.angle = Math.sin(angle2 * 0.498 - 0.5) * 0.34;
        arm10.angle = Math.sin(angle3) * 2.7181 / 4;
        arm2.x = arm.getEndX();
        arm2.y = arm.getEndY();
        arm3.x = arm2.getEndX() + 100 * Math.cos(counter);
        arm3.y = arm2.getEndY() + 100 * Math.sin(counter);
        arm5.x = arm4.getEndX();
        arm5.y = arm4.getEndY();
        arm6.x = arm5.getEndX() + 100 * Math.sin(counter);
        arm6.y = arm5.getEndY() + 100 * Math.abs(Math.sin(counter));
        arm8.x = arm7.getEndX();
        arm8.y = arm7.getEndY();
        arm9.x = arm8.getEndX();
        arm9.y = arm8.getEndY();
        arm10.x = arm9.getEndX() - 100 * Math.abs(Math.sin(0.5 * counter));
        arm10.y = arm9.getEndY() - 100 * Math.sin(0.5 * counter);
        angle += 0.05;//- (0.005/(1 + Math.abs(Math.sin(counter))));
        angle2 += 0.01;// - (0.005/(1 + Math.abs(Math.sin(counter))));
        angle3 += 0.027181;// - (0.005/(1 + Math.abs(Math.sin(counter))));
        /*arm.render(context);
        arm2.render(context);
        arm3.render(context);
        arm4.render(context);
        arm5.render(context);
        arm6.render(context);
        arm7.render(context);
        arm8.render(context);
        arm9.render(context);
        arm10.render(context);*/

        if (drawing) {
            context2.lineTo(arm3.getEndX(), arm3.getEndY());
            context2.stroke();
            context3.lineTo(arm6.getEndX(), arm6.getEndY());
            context3.stroke();
            context4.lineTo(arm10.getEndX(), arm10.getEndY());
            context4.stroke();
        }
        requestAnimationFrame(update);
    }
}
window.onload = function () {
    var canvas = document.getElementById("canvas"), canvas2 = document.getElementById("canvas2"),
        canvas3 = document.getElementById("canvas3"), context = canvas.getContext("2d"),
        context2 = canvas2.getContext("2d"), context3 = canvas3.getContext("2d"),
        width = canvas.width = canvas2.width = canvas3.width = window.innerWidth,
        height = canvas.height = canvas2.height = canvas3.height = window.innerHeight, drawing = true;


    var arm = Arm.create(width / 2, height / 2, 100, 0), angle = 0, angle2 = 0,
        arm2 = Arm.create(arm.getEndX(), arm.getEndY(), 100, 1.3),
        arm3 = Arm.create(arm2.getEndX(), arm2.getEndY(), 100, 1.3),
        arm4 = Arm.create(width / 2.1, height / 2.1, 100, -1)
    arm5 = Arm.create(arm4.getEndX(), arm4.getEndY(), 100, 0.3), arm6 = Arm.create(arm5.getEndX(), arm5.getEndY(), 100, 0.3),

        arm2.parent = arm;
    arm3.parent = arm2;
    arm4.parent = arm3;
    arm5.parent = arm4;
    arm6.parent = arm5;
    context2.lineWidth = 0.25;
    context3.lineWidth = 0.25

    update();
    document.body.addEventListener("click", function () {
        drawing = true;
    })

    function update() {
        if (drawing) {
            context2.beginPath();
            context2.moveTo(arm3.getEndX(), arm3.getEndY());
            context3.beginPath();
            context3.moveTo(arm6.getEndX(), arm6.getEndY());
        }

        context.clearRect(0, 0, width, height);
        arm.angle = Math.sin(angle) * 3.476;
        arm2.angle = Math.cos(angle * .502 + 2) * 3.92;
        arm3.angle = Math.sin(angle * 1.498 - 0.5) * 3.34;
        arm4.angle = Math.sin(angle2) * 2.476;
        arm5.angle = Math.cos(angle2 * .502 + 2) * 2.92;
        arm6.angle = Math.sin(angle2 * 1.498 - 0.5) * 2.34;
        arm2.x = arm.getEndX();
        arm2.y = arm.getEndY();
        arm3.x = arm2.getEndX();
        arm3.y = arm2.getEndY();
        arm5.x = arm4.getEndX();
        arm5.y = arm4.getEndY();
        arm6.x = arm5.getEndX();
        arm6.y = arm5.getEndY();
        angle += 0.05;
        angle2 += 0.1;
        arm.render(context);
        arm2.render(context);
        arm3.render(context);
        arm4.render(context);
        arm5.render(context);
        arm6.render(context);

        if (drawing) {
            context2.lineTo(arm3.getEndX(), arm3.getEndY());
            context2.stroke();
            context3.lineTo(arm6.getEndX(), arm6.getEndY());
            context3.stroke();
        }
        requestAnimationFrame(update);
    }
}
window.onload = function () {
    var canvas = document.getElementById("canvas"), canvas2 = document.getElementById("canvas2"),
        canvas3 = document.getElementById("canvas3"), context = canvas.getContext("2d"),
        context2 = canvas2.getContext("2d"), context3 = canvas3.getContext("2d"),
        width = canvas.width = canvas2.width = canvas3.width = window.innerWidth,
        height = canvas.height = canvas2.height = canvas3.height = window.innerHeight, drawing = true;


    var arm = Arm.create(width / 2, height / 2, 100, 0), angle = 0, angle2 = 0,
        arm2 = Arm.create(arm.getEndX(), arm.getEndY(), 100, 1.3),
        arm3 = Arm.create(arm2.getEndX(), arm2.getEndY(), 100, 1.3), arm4 = Arm.create(width / 4, height / 4, 100, -1)
    arm5 = Arm.create(arm4.getEndX(), arm4.getEndY(), 100, 0.3), arm6 = Arm.create(arm5.getEndX(), arm5.getEndY(), 100, 0.3),

        arm2.parent = arm;
    arm3.parent = arm2;
    arm4.parent = arm3;
    arm5.parent = arm4;
    arm6.parent = arm5;
    context2.lineWidth = 0.25;
    context3.lineWidth = 0.25

    update();
    document.body.addEventListener("click", function () {
        drawing = true;
    })

    function update() {
        if (drawing) {
            context2.beginPath();
            context2.moveTo(arm3.getEndX(), arm3.getEndY());
            context3.beginPath();
            context3.moveTo(arm6.getEndX(), arm6.getEndY());
        }

        context.clearRect(0, 0, width, height);
        arm.angle = Math.sin(angle) * 3.476;
        arm2.angle = Math.cos(angle * .502 + 2) * 3.92;
        arm3.angle = Math.sin(angle * 1.498 - 0.5) * 3.34;
        arm4.angle = Math.sin(angle2) * 2.476;
        arm5.angle = Math.cos(angle2 * .502 + 2) * 2.92;
        arm6.angle = Math.sin(angle2 * 1.498 - 0.5) * 2.34;
        arm2.x = arm.getEndX();
        arm2.y = arm.getEndY();
        arm3.x = arm2.getEndX();
        arm3.y = arm2.getEndY();
        arm5.x = arm4.getEndX();
        arm5.y = arm4.getEndY();
        arm6.x = arm5.getEndX();
        arm6.y = arm5.getEndY();
        angle += 0.05;
        angle2 += 0.1;
        arm.render(context);
        arm2.render(context);
        arm3.render(context);
        arm4.render(context);
        arm5.render(context);
        arm6.render(context);

        if (drawing) {
            context2.lineTo(arm3.getEndX(), arm3.getEndY());
            context2.stroke();
            context3.lineTo(arm6.getEndX(), arm6.getEndY());
            context3.stroke();
        }
        requestAnimationFrame(update);
    }
}

window.onload = function () {
    var canvas = document.getElementById("canvas"), context = canvas.getContext("2d"),
        width = canvas.width = window.innerWidth, height = canvas.height = window.innerHeight;

    context.translate(0, height / 2);
    context.scale(1, -1);

    for (var angle = 0; angle < Math.PI * 2; angle += .01) {
        var x = angle * 200, y = Math.sin(angle) * 200;

        context.fillStyle = "black";
        context.fillRect(x, y, 5, 5);

        y = Math.cos(angle) * 200;
        context.fillStyle = "red";
        context.fillRect(x, y, 5, 5);
    }
};/**
 * Created by cocodell on 5/2/2017.
 */

window.onload = function () {
    var canvas = document.getElementById("canvas"), context = canvas.getContext("2d"),
        width = canvas.width = window.innerWidth, height = canvas.height = window.innerHeight;

    var centerY = height * .5, centerX = width * .5, baseAlpha = 0.5, offset = 0.5, speed = 0.1, angle = 0;

    render();

    function render() {
        var alpha = baseAlpha + Math.sin(angle) * offset;

        context.fillStyle = "rgba(0, 0, 0, " + alpha + ")";

        context.clearRect(0, 0, width, height);
        context.beginPath();
        context.arc(centerX, centerY, 100, 0, Math.PI * 2, false);
        context.fill();

        angle += speed;

        requestAnimationFrame(render);
    }
};/**
 * Created by cocodell on 5/2/2017.
 */

window.onload = function () {
    var canvas = document.getElementById("canvas"), context = canvas.getContext("2d"),
        width = canvas.width = window.innerWidth, height = canvas.height = window.innerHeight, xres = 10, yres = 11;

    context.fillStyle = "black";
    context.fillRect(0, 0, width, height);
    context.fillStyle = "green";
    context.font = "12px Courier";
    context.translate(width / 2, height / 2);
    // context.scale(1.5, 1.5);
    // context.rotate(.1);
    context.transform(1.5, .3, 0.1, 1.5, 0, 0);

    for (var y = -height / 2; y < height / 2; y += yres) {
        for (var x = -width / 2; x < width / 2; x += xres) {
            var char = Math.random() < .5 ? "0" : "1";
            context.fillText(char, x, y);
        }
    }
};
/**
 * Created by cocodell on 5/2/2017.
 */
// Avoid `console` errors in browsers that lack a console.
(function () {
    var method;
    var noop = function () {
    };
    var methods = ['assert', 'clear', 'count', 'debug', 'dir', 'dirxml', 'error', 'exception', 'group', 'groupCollapsed', 'groupEnd', 'info', 'log', 'markTimeline', 'profile', 'profileEnd', 'table', 'time', 'timeEnd', 'timeline', 'timelineEnd', 'timeStamp', 'trace', 'warn'];
    var length = methods.length;
    var console = (window.console = window.console || {});

    while (length--) {
        method = methods[length];

        // Only stub undefined methods.
        if (!console[method]) {
            console[method] = noop;
        }
    }
}());

// Place any jQuery/helper plugins in here.
