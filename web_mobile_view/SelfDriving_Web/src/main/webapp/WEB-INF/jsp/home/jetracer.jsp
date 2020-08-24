<%@ page contentType="text/html; charset=UTF-8"%>
<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core"%>
<%@ taglib prefix="fmt" uri="http://java.sun.com/jsp/jstl/fmt"%>

<!DOCTYPE html>
<html>
  <head> 
  		<meta charset="utf-8">
		<meta http-equiv="X-UA-Compatible" content="IE=edge">
	    <title>AIOT FINAL PROJECT | TEAM 2</title>
	    <meta name="viewport" content="width=device-width, initial-scale=1">

		<script src="${pageContext.request.contextPath}/resource/jquery/jquery-3.4.1.min.js" crossorigin="anonymous"></script>
		<script src="${pageContext.request.contextPath}/resource/jquery-ui/jquery-ui.min.js"></script>
		<link rel="stylesheet" href="${pageContext.request.contextPath}/resource/jquery-ui/jquery-ui.min.css">
		
		<!--  Template 관련 설정 파일들 -->
	    <script src="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/popper.js/umd/popper.min.js"> </script>
	    <script src="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/bootstrap/js/bootstrap.min.js"></script>
	    <script src="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/jquery.cookie/jquery.cookie.js"> </script>
	    <script src="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/jquery-validation/jquery.validate.min.js"></script>
	    <script src="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/js/front.js"></script>
	    <script src="https://cdnjs.cloudflare.com/ajax/libs/paho-mqtt/1.0.1/mqttws31.min.js"></script>
	    
	    <link rel="stylesheet" href="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/bootstrap/css/bootstrap.min.css">
	    <link rel="stylesheet" href="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/font-awesome/css/font-awesome.min.css">
	    <link rel="stylesheet" href="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/css/font.css">
	    <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Muli:300,400,700">
		<link rel="stylesheet" href="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/css/style.default.css" id="theme-stylesheet">
	    <link rel=icon href="${pageContext.request.contextPath}/resource/img/jetracer.png">
	    <link rel="stylesheet" href="${pageContext.request.contextPath}/resource/css/yunjis.css">

		<script>
			$(function(){
				client = new Paho.MQTT.Client(location.hostname, 61614, new Date().getTime().toString());
				client.onMessageArrived = onMessageArrived;
				client.connect({onSuccess:onConnect});
				
				$(document).ready(function(){
				    $("#jet-racer1").load("${pageContext.request.contextPath}/home/1jet.do");
				    $("#jet-racer2").load("${pageContext.request.contextPath}/home/2jet.do");
				    $("#jet-racer3").load("${pageContext.request.contextPath}/home/3jet.do");
				});
			});
			
			function onConnect() {
				console.log("mqtt broker connected")
				client.subscribe("/1jetracer");
				client.subscribe("/2jetracer");
				client.subscribe("/3jetracer");
				client.subscribe("/mirror");
			}
			
			function onMessageArrived(message) {
				if(message.destinationName =="/mirror") {
					const json = message.payloadString;
 					const obj = JSON.parse(json);
					$("#motionView").attr("src", "data:image/jpg;base64,"+ obj.Cam);
					
					handMotion = obj.hands;
					console.log(handMotion);
					
					if(handMotion=='rock'){
						$("#motionName1").text('STOP');
						$("#targetSpeed1").text("Speed : 0");
						
						message = new Paho.MQTT.Message("speed:"+ 0);
						message.destinationName = "/1motion/stop";
						message.qos = 0;
						client.send(message);
					}
					
					else if(handMotion=='5fingers'){
						$("#motionName1").text('GO');
						$("#targetSpeed1").text("Speed : 60");
						
						message = new Paho.MQTT.Message("speed:"+ 60);
						message.destinationName = "/1motion/go";
						message.qos = 0;
						client.send(message);
					}
					
					else if(handMotion=='unlabeled'){
						$("#motionName1").text('---');
						$("#targetSpeed1").text("Speed : ---");
					}

					if(obj.direction=="left"){
						location.href="main.do";
					}
					
					else if (obj.direction=="right"){
						location.href="history.do";
					}

				}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////		jetracer #1 	/////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 				if(message.destinationName =="/1jetracer") {
 					const json = message.payloadString;
 					const obj = JSON.parse(json);
 					
 					$("#jetView1").attr("src", "data:image/jpg;base64,"+ obj.Cam);
					
				/////////////////////////////////////////////////		배터리 상태		///////////////////////////////////////////////////////////////////////
					bat1 = obj.battery;
			      	bat1 = parseInt(bat1);
					//console.log("battery1:",obj.battery, "%");
			      	
			      	if (bat1 >= 100){
			      		bat1 == 100;
			      		document.getElementById('batt').style.backgroundColor = 'dimgray';
			      		document.getElementById('batt').style.opacity = '0.9';
			      		document.getElementById('batt').style.color = 'white';
			      		document.getElementById('adtt').style.backgroundColor = 'red';
			      		document.getElementById('jet1Battery').style.width = bat1 + '%';
			      		$("#batt").attr("value", "Battery");
			      		$("#jetRacerText1").text("100%");
			      	}
			      	else if (40<bat1 && bat1<100){
			      		document.getElementById('batt').style.backgroundColor = '#ADFF2F';
			      		document.getElementById('batt').style.color = 'black';
			      		document.getElementById('adtt').style.backgroundColor = 'dimgray';
			      		document.getElementById('adtt').style.opacity = '0.9';
			      		document.getElementById('jet1Battery').style.width = String(bat1) + '%';
			      		$("#batt").attr("value", "Battery");
			      		$("#jetRacerText1").text(String(bat1) + "%");
			      	}
			      	else if (bat1 <= 40){
			      		document.getElementById('batt').style.backgroundColor = 'red';
			      		document.getElementById('batt').style.color = 'white';
			      		document.getElementById('adtt').style.backgroundColor = 'dimgray';
			      		document.getElementById('adtt').style.opacity = '0.9';
			      		document.getElementById('vacant').style.backgroundColor ="transparent";
			      		document.getElementById('jet1Battery').style.width = String(bat1) + '%';
			      		$("#batt").attr("value", "CHARGE NOW !!!");
			      		$("#jetRacerText1").text(String(bat1) + "%");
			      	}
				/////////////////////////////////////////////////		dc speed		///////////////////////////////////////////////////////////////////
					speed1 = obj.speed;
			      	speed1=Math.round(speed1);
			      	//console.log("speed:", speed1);
			      	if(speed1 < 40){
			      		$("#MotorSpeed").text(0 +" km/h");
			      	}
			      	else if(40 <= speed1){
			      		$("#MotorSpeed").text(speed1 +" km/h");
			      	}
				/////////////////////////////////////////////////		주행 구역		///////////////////////////////////////////////////////////////////////
			      	//area1 = obj.area;
			   		var text = "";
					var possible = "ABCDEFGHIJKLMNOPQRST";
					text = possible.charAt(Math.floor(Math.random() * possible.length));
			      	$("#district1").text("Zone " + text);

				/////////////////////////////////		외부 온도		///////////////////////////////////////////////////////////////////////
			      	//temp1 = obj.temp;
			      	temp1 = parseInt(((Math.random()*10+1))+20);
			      	$("#Temperature").text(temp1 +" °C");
				}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////		jetracer #2		/////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
				if(message.destinationName =="/2jetracer") {
					const json = message.payloadString;
 					const obj = JSON.parse(json);
					$("#jetView2").attr("src", "data:image/jpg;base64,"+ obj.Cam);
					
					/////////////////////////////////////////////////		배터리 상태		///////////////////////////////////////////////////////////////////////

					bat2 = obj.battery;
			      	bat2 = parseInt(bat2);
			      	
			      	servo2 = obj.servo;
			      	console.log("servo:" + servo2);
			      	
			      	if (bat2 >= 100){
			      		bat2 == 100;
			      		document.getElementById('batt2').style.backgroundColor = 'dimgray';
			      		document.getElementById('batt2').style.opacity = '0.9';
			      		document.getElementById('batt2').style.color = 'white';
			      		document.getElementById('adtt2').style.backgroundColor = 'red';
			      		document.getElementById('jet2Battery').style.width = bat2 + '%';
			      		$("#batt2").attr("value", "Battery");
			      		$("#jetRacerText2").text("100%");
			      	}
			      	else if (40<bat2 && bat2<100){
			      		document.getElementById('batt2').style.backgroundColor = '#ADFF2F';
			      		document.getElementById('batt2').style.color = 'black';
			      		document.getElementById('adtt2').style.backgroundColor = 'dimgray';
			      		document.getElementById('adtt2').style.opacity = '0.9';
			      		document.getElementById('jet2Battery').style.width = String(bat2) + '%';
			      		$("#batt2").attr("value", "Battery");
			      		$("#jetRacerText2").text(String(bat2) + "%");
			      	}
			      	else if (bat2 <= 40){
			      		document.getElementById('batt2').style.backgroundColor = 'red';
			      		document.getElementById('batt2').style.color = 'white';
			      		document.getElementById('adtt2').style.backgroundColor = 'dimgray';
			      		document.getElementById('adtt2').style.opacity = '0.9';
			      		document.getElementById('vacant2').style.backgroundColor ="transparent";
			      		document.getElementById('jet2Battery').style.width = String(bat2) + '%';
			      		$("#batt2").attr("value", "CHARGE NOW !!!");
			      		$("#jetRacerText2").text(String(bat2) + "%");
			      	}
	
				/////////////////////////////////////////////////		dc speed		///////////////////////////////////////////////////////////////////
					speed2 = obj.speed;
			      	speed2=Math.round(speed2);
			      	
			      	if(speed2 < 40){
			      		$("#MotorSpeed2").text(0 +" km/h");
			      		//$("#MotorSpeed").attr("value", 0 +" km/h");
			      	}
			      	else if(40 <= speed2){
			      		$("#MotorSpeed2").text(speed2 +" km/h");
			      		//$("#MotorSpeed").attr("value", speed1 +" km/h");
			      	}
			      	//console.log("speed:", speed1);

				/////////////////////////////////////////////////		주행 구역		///////////////////////////////////////////////////////////////////////
			      	//area1 = obj.area;
			   		var text2 = "";
					var possible = "ABCDEFGHIJKLMNOPQRST";
					text2 = possible.charAt(Math.floor(Math.random() * possible.length));
					//console.log(text);
			      	$("#district2").text("Zone " + text2);
			      	//document.getElementById('jet1District').style.width = (area1*5) + '%';

				/////////////////////////////////////////////////		외부 온도		///////////////////////////////////////////////////////////////////////
			      	//temp1 = obj.temp;
			      	temp2 = parseInt(((Math.random()*10+1))+20);
			      	$("#Temperature2").text(temp2 +" °C");
			      	//$("#Temperature").attr("value", temp1 +" °C");
				}
				

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////		jetracer #3		/////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

				if(message.destinationName =="/3jetracer") {
					const json = message.payloadString;
					const obj = JSON.parse(json);
					$("#jetView3").attr("src", "data:image/jpg;base64,"+ obj.Cam);
				
					console.log("1:뱉"+obj.battery);
					console.log("2:섭"+obj.servo);
					console.log("3:슾"+obj.speed);
					console.log("4:랍"+obj.label);
					console.log("5:밗"+obj.boxes);
					console.log("6:렢"+obj.line_left);
					console.log("7:뢑"+obj.line_right);

 					bat3 = obj.battery;
 					bat3 = parseInt(bat3);

			      	if (bat3 >= 100){
			      		bat3 == 100;
			      		document.getElementById('batt3').style.backgroundColor = 'dimgray';
			      		document.getElementById('batt3').style.opacity = '0.9';
			      		document.getElementById('batt3').style.color = 'white';
			      		document.getElementById('adtt3').style.backgroundColor = 'red';
			      		document.getElementById('jet3Battery').style.width = bat3 + '%';
			      		$("#batt3").attr("value", "Battery");
			      		$("#jetRacerText3").text("100%");
			      	}
			      	else if (40<bat3 && bat3<100){
			      		document.getElementById('batt3').style.backgroundColor = '#ADFF2F';
			      		document.getElementById('batt3').style.color = 'black';
			      		document.getElementById('adtt3').style.backgroundColor = 'dimgray';
			      		document.getElementById('adtt3').style.opacity = '0.9';
			      		document.getElementById('jet3Battery').style.width = String(bat3) + '%';
			      		$("#batt3").attr("value", "Battery");
			      		$("#jetRacerText3").text(String(bat3) + "%");
			      	}
			      	else if (bat3 <= 40){
			      		document.getElementById('batt3').style.backgroundColor = 'red';
			      		document.getElementById('batt3').style.color = 'white';
			      		document.getElementById('adtt3').style.backgroundColor = 'dimgray';
			      		document.getElementById('adtt3').style.opacity = '0.9';
			      		document.getElementById('vacant3').style.backgroundColor ="transparent";
			      		document.getElementById('jet3Battery').style.width = String(bat3) + '%';
			      		$("#batt3").attr("value", "CHARGE NOW !!!");
			      		$("#jetRacerText3").text(String(bat3) + "%");
			      	}
	
				/////////////////////////////////////////////////		dc speed		///////////////////////////////////////////////////////////////////
					speed3 = obj.speed;

			      	if(speed3 < 40){
			      		$("#MotorSpeed3").text(0 +" km/h");
			      	}
			      	else if(40 <= speed3){
			      		$("#MotorSpeed3").text(speed3 +" km/h");
			      	}

				/////////////////////////////////////////////////		주행 구역		///////////////////////////////////////////////////////////////////////
			      	//area1 = obj.area;
			   		var text3 = "";
					var possible = "ABCDEFGHIJKLMNOPQRST";
					text3 = possible.charAt(Math.floor(Math.random() * possible.length));
			      	$("#district3").text("Zone " + text3);
			      	
				/////////////////////////////////////////////////		외부 온도		///////////////////////////////////////////////////////////////////////
			      	temp3 = parseInt(((Math.random()*10+1))+20);
			      	$("#Temperature3").text(temp3 +" °C");
				}
			}
 			
			/////////////////////////////////////////////////		매뉴얼 컨트롤 모드 변경		///////////////////////////////////////////////////////////////////////						
			function manual(value, key1, key2, key3){
				console.log(value);
				
				if (value == 'On'){
					console.log("메뉴얼 1 실행");
					console.log(value);
					
					document.getElementById('modeOn').style.backgroundColor = '#ADFF2F';
					document.getElementById('modeOn').style.color = 'black';
					document.getElementById('modeOff').style.backgroundColor = 'dimgray';
					document.getElementById('modeOff').style.color = 'white';
					document.getElementById('manual_control').style.display = 'block';
					
					alert("Converting to Manual Driving Mode-1");

					$("#jet-racer1").keydown(function(event) {
						if (event.keyCode == '38') {
						  	console.log("달리자")
		 				  	message = new Paho.MQTT.Message("speed:"+ 40);
							message.destinationName = "/1manual/go";
							message.qos = 0;
							client.send(message);
						}
						if (event.keyCode == '40') { 		
						    console.log("멈추거라")
		 				  	message = new Paho.MQTT.Message("speed:" + 0);
							message.destinationName = "/1manual/stop";
							message.qos = 0;
							client.send(message);
						}
						if (event.keyCode == '37'){
						  	console.log("toL1");
		 				  	message = new Paho.MQTT.Message("lineChangeToL");
							message.destinationName = "/1manual/toL";
							message.qos = 0;
							client.send(message);
							console.log(message);
						}
						if (event.keyCode == '39'){ 	
							console.log("toR1");
		 				  	message = new Paho.MQTT.Message("lineChangeToR");
							message.destinationName = "/1manual/toR";
							message.qos = 0;
							client.send(message);
						}
					});
				}
				
				else if (value == 'Off'){
					console.log("메뉴얼 1 종료");
					console.log(value);
					
					document.getElementById('modeOn').style.backgroundColor = 'dimgray';
					document.getElementById('modeOn').style.color = 'white';
					document.getElementById('modeOff').style.backgroundColor = '#ADFF2F';
					document.getElementById('modeOff').style.color = 'black';
					document.getElementById('manual_control').style.display = 'none';
					alert("Converting to Auto Driving Mode-1");
					
					message = new Paho.MQTT.Message(value);
					message.destinationName = "/1manual/mode";
					message.qos = 0;
					client.send(message);
				}
			}
			
			function manual2(value){
				console.log("메뉴얼 2 실행해보자");
				console.log(value);

				if (value == 'On'){
					document.getElementById('modeOn2').style.backgroundColor = '#ADFF2F';
					document.getElementById('modeOn2').style.color = 'black';
					document.getElementById('modeOff2').style.backgroundColor = 'dimgray';
					document.getElementById('modeOff2').style.color = 'white';
					document.getElementById('manual_control2').style.display = 'block';
					
					alert("Converting to Manual Driving Mode-2");
					
					$("#jet-racer2").keydown(function(event) {
						if (event.keyCode == '38'){
						  	console.log("달리자2");
		 				  	message = new Paho.MQTT.Message("speed");
							message.destinationName = "/2manual/go";
							message.qos = 0;
							client.send(message);
							console.log(message);
						}
						if (event.keyCode == '40'){ 	
							console.log("멈추자2");
		 				  	message = new Paho.MQTT.Message("speed");
							message.destinationName = "/2manual/stop";
							message.qos = 0;
							client.send(message);
						}
						if (event.keyCode == '37'){
						  	console.log("toL2");
		 				  	message = new Paho.MQTT.Message("lineChangeToL");
							message.destinationName = "/2manual/toL";
							message.qos = 0;
							client.send(message);
							console.log(message);
						}
						if (event.keyCode == '39'){ 	
							console.log("toR2");
		 				  	message = new Paho.MQTT.Message("lineChangeToR");
							message.destinationName = "/2manual/toR";
							message.qos = 0;
							client.send(message);
						}
					});
				}
			
				else if (value == 'Off'){
					console.log("메뉴얼 2 종료");
					console.log(value);
					
					document.getElementById('modeOn2').style.backgroundColor = 'dimgray';
					document.getElementById('modeOn2').style.color = 'white';
					document.getElementById('modeOff2').style.backgroundColor = '#ADFF2F';
					document.getElementById('modeOff2').style.color = 'black';
					document.getElementById('manual_control2').style.display = 'none';
					alert("Converting to Auto Driving Mode-2");

					message = new Paho.MQTT.Message(value);
					message.destinationName = "/2manual/mode";
					message.qos = 0;
					client.send(message);
				}
			};
			
			function manual3(value){
				console.log("메뉴얼 3 실행해보자");
				console.log(value);
				
				if (value == 'On'){
					document.getElementById('modeOn3').style.backgroundColor = '#ADFF2F';
					document.getElementById('modeOn3').style.color = 'black';
					document.getElementById('modeOff3').style.backgroundColor = 'dimgray';
					document.getElementById('modeOff3').style.color = 'white';
					document.getElementById('manual_control3').style.display = 'block';

					alert("Converting to Manual Driving Mode");
					
					$("#jet-racer3").keydown(function(event) {
						if (event.keyCode == '38') {
						  	console.log("달리자3")
		 				  	message = new Paho.MQTT.Message("speed:"+ 40);
							message.destinationName = "/3manual/go";
							message.qos = 0;
							client.send(message);
						}
					
						if (event.keyCode == '40') { 		
						    console.log("멈추거라3")
		 				  	message = new Paho.MQTT.Message("speed:" + 0);
							message.destinationName = "/3manual/stop";
							message.qos = 0;
							client.send(message);
						}
						if (event.keyCode == '37'){
						  	console.log("toL3");
		 				  	message = new Paho.MQTT.Message("lineChangeToL");
							message.destinationName = "/3manual/toL";
							message.qos = 0;
							client.send(message);
							console.log(message);
						}
						if (event.keyCode == '39'){ 	
							console.log("toR3");
		 				  	message = new Paho.MQTT.Message("lineChangeToR");
							message.destinationName = "/3manual/toR";
							message.qos = 0;
							client.send(message);
						}
					});
				}
				
				else if (value == 'Off'){
					console.log("메뉴얼 3 종료");
					console.log(value);
					
					document.getElementById('modeOn3').style.backgroundColor = 'dimgray';
					document.getElementById('modeOn3').style.color = 'white';
					document.getElementById('modeOff3').style.backgroundColor = '#ADFF2F';
					document.getElementById('modeOff3').style.color = 'black';
					document.getElementById('manual_control3').style.display = 'none';
					alert("Converting to Auto Driving Mode");

					message = new Paho.MQTT.Message(value);
					message.destinationName = "/3manual/mode";
					message.qos = 0;
					client.send(message);
				}
			}
		</script>
	</head>
	
	<body>
		<header class="header">   
	      <nav class="navbar navbar-expand-lg" style="height: 50px">
	        <div class="container-fluid d-flex align-items-center justify-content-between">
	          <div class="navbar-header">
	            <a href="${pageContext.request.contextPath}/home/main.do" class="navbar-brand">
		              <div class="brand-text brand-big visible text-uppercase" style="font-size: x-large"><strong class="text-primary">AIOT</strong><strong>Admin</strong></div>
		              <div class="brand-text brand-sm"><strong class="text-primary">A</strong><strong>A</strong></div>
		         </a>
	            <button class="sidebar-toggle"><i class="fa fa-long-arrow-left"></i></button>
	          </div>
	          <div class="right-menu list-inline no-margin-bottom">    
	            <div class="list-inline-item logout"><a id="logout" href="${pageContext.request.contextPath}/home/intro.do" class="nav-link"> <span class="d-none d-sm-inline">to INTRO </span><i class="icon-logout"></i></a></div>
	          </div>
	        </div>
	      </nav>
	    </header>
	    
		<div class="d-flex align-items-stretch" style="height: 875px;">
	      <nav id="sidebar" style="height: 1030px;">
	        <div class="sidebar-header d-flex align-items-center">
	          <div class="avatar" style="width: 100px; height: 100px; align-itself: center"><img src="${pageContext.request.contextPath}/resource/img/milk.jpg" class="img-fluid rounded-circle"></div>
	          <div class="title">
	            <h1 class="h5" style="color: lightgray">AIoT Project</h1>
	            <p style="color: lightgray">관리자</p>
	          </div>
	        </div>
	        <span class="heading" style="color:#DB6574">CATEGORIES</span>
	        <ul class="list-unstyled">
	          <li><a href="${pageContext.request.contextPath}/home/main.do" style="color: lightgray"> <i class="icon-home"></i>메인 페이지</a></li>
	          <li class="active"><a href="${pageContext.request.contextPath}/home/jetracer.do" style="color: lightgray"> <i class="icon-writing-whiteboard"></i>탐지봇 현황 </a></li>
	          <li><a href="${pageContext.request.contextPath}/home/history.do" style="color: lightgray"> <i class="icon-grid"></i>탐지 히스토리 조회 </a></li>
	          <li><a href="${pageContext.request.contextPath}/home/status.do" style="color: lightgray"> <i class="icon-padnote"></i>실시간 탐지 | 대응 현황</a></li>
	          <li><a href="${pageContext.request.contextPath}/home/analysis.do" style="color: lightgray"> <i class="icon-chart"></i>탐지 결과 분석</a></li>
	       	</ul>
	      </nav>
	      
	      <div class="page-content" style="top: -50px; height: 1080px; padding-bottom: 0px; ">
	      
			<div style="margin-top: 65px">
			  <ul class="nav nav-pills mb-3" id="pills-tab" role="tablist" style="max-width: 100%; margin-left: 30px" >
				  <li class="nav-item" role="presentation" style="width: 33%; text-align: center">
				    <a class="nav-link active" id="jet-racer1-tab" data-toggle="pill" href="#jet-racer1" role="tab" aria-controls="jet-racer1" aria-selected="true" style="font-weight: bold; font-size: large; ">Jet-Racer #1</a>
				  </li>
				  <li class="nav-item" role="presentation" style="width: 33%; text-align: center">
				    <a class="nav-link" id="jet-racer2-tab" data-toggle="pill" href="#jet-racer2" role="tab" aria-controls="jet-racer2" aria-selected="false" style="font-weight: bold; font-size: large;">Jet-Racer #2</a>
				  </li>
				  <li class="nav-item" role="presentation" style="width: 33%; text-align: center">
				    <a class="nav-link" id="jet-racer3-tab" data-toggle="pill" href="#jet-racer3" role="tab" aria-controls="jet-racer3" aria-selected="false" style="font-weight: bold; font-size: large; ">Jet-Racer #3</a>
				  </li>
		      </ul>
	      	</div>

			<div class="tab-content" id="pills-tabContent" style="height: 944px; margin-left: 25px; margin-right: 25px;  border-color: dimgray; border-style:solid; border-width:medium; ">
			  
		  	<!-- jet racer # 1 -->
		  	<div class="tab-pane fade show active" id="jet-racer1" role="tabpanel" aria-labelledby="jet-racer1-tab"></div> 
		 		
			<!-- jet racer # 2 -->
			<div class="tab-pane fade" id="jet-racer2" role="tabpanel" aria-labelledby="jet-racer2-tab"></div>
			  
			<!-- jet racer # 3 -->
			<div class="tab-pane fade" id="jet-racer3" role="tabpanel" aria-labelledby="jet-racer3-tab"></div>
		</div>
	</div>
	</div>
	</body>
</html>