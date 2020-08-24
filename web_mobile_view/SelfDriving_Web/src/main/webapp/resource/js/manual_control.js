$(function(){
	client = new Paho.MQTT.Client(location.hostname, 61614, new Date().getTime().toString());
	client.onMessageArrived = onMessageArrived;
	client.connect({onSuccess:onConnect});
});

function onConnect() {
	console.log("mqtt broker connected");
}

var speed = 0;
var order = "stop";
var stopped = false;
function backTire_control(setOrder, setSpeed) {
	var message = 0;
	if(setSpeed != undefined) {
		speed = setSpeed;
	}
	if(stopped == true) {
		speed = 0;
		stopped = false;
	}
	if(setOrder != '0') {
		order = setOrder;
	}
	if(order == "stop") {
		speed = 0;
	}
	
	var target = {
			direction:order,
			pwm:speed
	};
	
	message = new Paho.MQTT.Message(JSON.stringify(target));
	message.destinationName = "command/backTire/button";
	client.send(message);
}

function rgbLed_red(){
	console.log("rgbRED");
	message = new Paho.MQTT.Message("red")
	message.destinationName = "command/rgbLed/red";
	client.send(message);
}

function rgbLed_green(){
	message = new Paho.MQTT.Message("green")
	message.destinationName = "command/rgbLed/green";
	client.send(message);
}

function rgbLed_blue(){
	message = new Paho.MQTT.Message("blue")
	message.destinationName = "command/rgbLed/blue";
	client.send(message);
}

function rgbLed_off(){
	message = new Paho.MQTT.Message("off")
	message.destinationName = "command/rgbLed/off";
	client.send(message);
}

var isPressed = false;

document.onkeydown = onkeydown_handler;
document.onkeyup = onkeyup_handler;

function onkeydown_handler(event) {
	var keycode = event.which || event.keycode;
	console.log(keycode);
	if(keycode == 87 || keycode== 65 || keycode== 83 || keycode== 68){		//카메라 제어
		if(keycode == 83){					//앞
			$("#cameradown").css("background-color", "#FFF8DC");
			var topic="command/camera/front";
		} else if(keycode == 65){			//왼쪽
			$("#cameraleft").css("background-color", "#FFF8DC");
			console.log(keycode);
			var topic="command/camera/left";
		} else if(keycode == 87){			//뒤
			$("#cameraup").css("background-color", "#FFF8DC");
			var topic="command/camera/back";
		} else if(keycode == 68){			//오른쪽
			$("#cameraright").css("background-color", "#FFF8DC");
			var topic="command/camera/right";
		}
		message = new Paho.MQTT.Message("camera");
		message.destinationName = topic;
		client.send(message);
	}
	if(keycode == 37 || keycode == 39) {
		if(keycode == 37) {
			//left
			$("#left").css("background-color", "#FFF8DC");
			var topic = "command/frontTire/left";
			console.log(topic);
		}else if(keycode == 39) {
			//right
			$("#right").css("background-color", "#FFF8DC");
			topic = "command/frontTire/right";
			console.log(topic);
		}
		message = new Paho.MQTT.Message("frontTire");
		message.destinationName = topic;
		client.send(message);
	}
	if(keycode == 38 || keycode == 40 || keycode == 32) {
		if(keycode == 38) {
			// up
			$("#up").css("background-color", "#FFF8DC");
			var topic = "command/backTire/forward";
		} else if(keycode == 40) {
			// down
			$("#down").css("background-color", "#FFF8DC");
			var topic = "command/backTire/backward";
		} else if(keycode == 32) {
			//spacebar
			stopped = true;
			var topic = "command/backTire/stop";
		}
		message = new Paho.MQTT.Message("backTire");
		message.destinationName = topic;
		client.send(message);
	}
	if(keycode == 100 || keycode == 102) {		// 거리 센서 제어
		if(keycode == 100) {					//좌
			$("#sonicleft").css("background-color", "#FFF8DC");
			var topic = "command/distance/left";
		} else if(keycode == 102) {				//우
			$("#sonicright").css("background-color", "#FFF8DC");
			var topic = "command/distance/right";
		} 
		message = new Paho.MQTT.Message("distance");
		message.destinationName = topic;
		client.send(message);
	}
}

function onkeyup_handler(event) {
	var keycode = event.which || event.keycode;
	if(keycode == 37 || keycode == 39) {
		$("#left").css("background-color", "");
		$("#right").css("background-color", "");
		var topic = "command/frontTire/front";
		message = new Paho.MQTT.Message("frontTire");
		message.destinationName = topic;
		client.send(message);
	}
	if(keycode == 38 || keycode == 40) {
		$("#up").css("background-color", "");
		$("#down").css("background-color", "");
		var topic = "command/backTire/respeed";
		message = new Paho.MQTT.Message("backTire");
		message.destinationName = topic;
		client.send(message);
	}
	if(keycode == 87 || keycode== 65 || keycode== 83 || keycode== 68) {
		$("#cameraup").css("background-color", "");
		$("#cameradown").css("background-color", "");
		$("#cameraright").css("background-color", "");
		$("#cameraleft").css("background-color", "");
	}
	if(keycode == 100 || keycode == 102) {
		$("#sonicleft").css("background-color", "");
		$("#sonicright").css("background-color", "");
	}
}

var backbuttonPressed = false;
var frontbuttonPressed = false;
function tire_button_down(direction) {
	if(direction=="up" || direction=="down") {
		backbuttonPressed = true;
	}
	else {
		frontbuttonPressed = true;
	}
	
	tire_control(direction);
	
}

function tire_button_up(direction) {
	if(direction=="up" || direction=="down") {
		backbuttonPressed = false;
		var topic = "command/backTire/respeed";
		message = new Paho.MQTT.Message("backTire");
		message.destinationName = topic;
		client.send(message);
	} else {
		frontbuttonPressed = false;
		var topic = "command/frontTire/front";
		message = new Paho.MQTT.Message("frontTire");
		message.destinationName = topic;
		client.send(message);
	}
}

function tire_control(direction) {
	if(backbuttonPressed) {
		if(direction == 'up') {
			var topic = "command/backTire/forward";
		}else if(direction == 'down') {
			var topic = "command/backTire/backward";
		}
		message = new Paho.MQTT.Message("tire");
		message.destinationName = topic;
		client.send(message);
		
		setTimeout(function() {
			tire_control(direction);
		}, 30);
	}
	if(frontbuttonPressed) {
		if(direction == 'left') {
			var topic = "command/frontTire/left";
		}else if(direction == 'right') {
			var topic = "command/frontTire/right";
		}
		message = new Paho.MQTT.Message("tire");
		message.destinationName = topic;
		client.send(message);
		
		setTimeout(function() {
			tire_control(direction);
		}, 30);
	}
}


$(function() {
	var directions = document.querySelectorAll("#tire_control a");
	directions.forEach(function(a) {
		console.log(a.id);
		a.addEventListener("touchstart", function() {
			go = true;
			tire_control_touch(a.id);
		});
		a.addEventListener("touchend", function() {
			go = false;
		});
	});
});

var go = false;
function tire_control_touch(direction) {
	if(go) {
		if(direction == "up") {
			var topic = "command/backTire/forward";
		}
		if(direction == "down") {
			var topic = "command/backTire/backward";
		}
		if(direction == "left") {
			var topic = "command/frontTire/left";
		}
		if(direction == "right") {
			var topic = "command/frontTire/right";
		}
		message = new Paho.MQTT.Message("tire");
		message.destinationName = topic;
		client.send(message);
		setTimeout(function() {
			tire_control_touch(direction);
		}, 30);
	}else {
		if(direction == "up" || direction == "down") {
			var topic = "command/backTire/respeed";
			message = new Paho.MQTT.Message("tire");
			message.destinationName = topic;
			client.send(message);	
		}else {
			var topic = "command/frontTire/front";
			message = new Paho.MQTT.Message("tire");
			message.destinationName = topic;
			client.send(message);	
		}
		
	}	
}
	

$(function() {
	var motor_direction = document.querySelectorAll("#motor_control a");
	motor_direction.forEach(function(a) {
		a.addEventListener("touchstart", function() {
			move = true;
			camera_control_touch(a.id);
		});
		a.addEventListener("touchend", function() {
			move = false;
		});
	});
	
	
});
var move = false;
function camera_control_touch(motor_direction){
	if(move){
		if(motor_direction == 'cameraup') {
			var topic="command/camera/back";
		}if(motor_direction == 'cameradown') {
			var topic="command/camera/front";
		}if(motor_direction == 'cameraleft') {
			var topic="command/camera/left";
		}if(motor_direction == 'cameraright') {
			var topic="command/camera/right";
		}if(motor_direction == 'sonicleft') {
			var topic="command/distance/left";
		}if(motor_direction == 'sonicright') {
			var topic="command/distance/right";
		}
		message = new Paho.MQTT.Message("camera&sonic");
		message.destinationName = topic;
		client.send(message);
		setTimeout(function() {
			camera_control_touch(motor_direction);
		}, 30);
	}
}

function click_w() {
  console.log("클릭 w 실행됨");
  message = new Paho.MQTT.Message("click_w")
  message.destinationName = "command/camera/back";
  client.send(message);
}
function click_a() {
  console.log("클릭 a 실행됨");
  message = new Paho.MQTT.Message("click_a")
  message.destinationName = "command/camera/left";
  client.send(message);
}
function click_d() {
  message = new Paho.MQTT.Message("click_d")
  message.destinationName = "command/camera/right";
  client.send(message);
}
function click_s() {
  message = new Paho.MQTT.Message("click_s")
  message.destinationName = "command/camera/front";
  client.send(message);
}

function click_up() {
  message = new Paho.MQTT.Message("click_s")
  message.destinationName = "command/backTire/forward";
  client.send(message);
}
function click_down() {
  message = new Paho.MQTT.Message("click_s")
  message.destinationName = "command/backTire/backward";
  client.send(message);
}
function click_left() {
  message = new Paho.MQTT.Message("click_left")
  message.destinationName = "command/frontTire/left";
  client.send(message);
}
function click_right() {
  message = new Paho.MQTT.Message("click_right")
  message.destinationName = "command/frontTire/right";
  client.send(message);
}

function click_4() {
  message = new Paho.MQTT.Message("click_s")
  message.destinationName = "command/distance/left";
  client.send(message);
}
function click_6() {
  message = new Paho.MQTT.Message("click_s")
  message.destinationName = "command/distance/right";
  client.send(message);
}

function camera_capture() {
	message = new Paho.MQTT.Message("capture")
	message.destinationName = "command/camera/capture";
	client.send(message);
}