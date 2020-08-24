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

		
	</head>
	
	<body>
		      	<div id=showView style="margin-top: 30px; margin-left: 30px">
             		<div id="title" style="background-color: #864DD9; width: 490px; color: white; font-weight: bold;justify-content: center;">Auto Driving Situation</div>
             		<div id="vacant2" style="background-color: transparent; width: 5px"></div>
             		<div id=batteryMode>
	              		<div id="batMode" style="width: 190px">
	              			<input id="batt2" value="Battery" readonly="readonly" style="border-color: transparent; background-color: #ADFF2F; text-align: center; color: black; width: 190px; font-weight: bold;justify-content: center;">
	              		</div>
						<div id="adtMode2" style="width: 190px">
							<input id="adtt2" value="Adapter Connected" readonly="readonly" style="border-color: transparent; background-color:dimgray; text-align: center; color: white; width: 190px; font-weight: bold;justify-content: center;">
						</div>
              		</div>
             	</div>
             	
             	<img id=jetView2 style="width: 490px; height: 370px; padding-left: 0px; padding-right: 0px; margin-left: 30px; margin-top: 0px"/>
             	
		      	<section class="no-padding-top no-padding-bottom" style="top:187px; position: absolute">
		          <div class="container-fluid">
		            <div class="row">
		              <div class="col-md-3 col-sm-6">
		                <div class="statistic-block block" style="width: 380px; height: 100px; margin-bottom: 10px; padding-bottom: 0px; margin-left: 495px">
		                  <div class="progress-details d-flex align-items-end justify-content-between">
		                    <div class="title">
		                      <div class="icon"><i class="icon-writing-whiteboard"></i></div><strong style="color: white">Battery Status</strong>
		                    </div> 
		                    <div class="number dashtext-1" id="jetRacerText2">
		                    	100%
		                    </div>
		                  </div>
		                  <div class="progress progress-template">
		                    <div id="jet2Battery" role="progressbar" style="width: 100%" aria-valuenow="30" aria-valuemin="0" aria-valuemax="100" class="progress-bar progress-bar-template dashbg-1"></div>
		                  </div>
		                </div>
		                <div class="statistic-block block" style="width: 380px; height: 80px; margin-left: 495px; justify-content: center; margin-bottom: 10px; padding-bottom: 0px">
		                  <div class="progress-details d-flex align-items-end justify-content-between" style="justify-content: center;">
		                    <div class="title" style="padding-bottom: 30px">
		                      <div class="icon"><i class="icon-info"></i></div><strong style="color: white">Driving District</strong>
		                    </div>
		                    <div class="number dashtext-1" id="district2" style="padding-bottom: 30px">
		                    	Zone A
		                    </div>
		                 </div>
		                </div>
 		                <div class="statistic-block block" style="width: 380px; height: 80px; margin-left: 495px; justify-content: center; margin-bottom: 10px; padding-bottom: 0px">
		                  <div class="progress-details d-flex align-items-end justify-content-between" style="justify-content: center;">
		                    <div class="title" style="padding-bottom: 30px">
		                      <div class="icon"><i class="icon-info"></i></div><strong style="color: white">Motor Speed</strong>
		                    </div>
		                    <div class="number dashtext-1" id="MotorSpeed2" style="padding-bottom: 30px">
		                    	60 km/h
		                    </div>
		                 </div>
		                </div>
		                <div class="statistic-block block" style="width: 380px; height: 80px; margin-left: 495px; justify-content: center; margin-bottom: 10px; padding-bottom: 0px">
		                  <div class="progress-details d-flex align-items-end justify-content-between" style="justify-content: center;">
		                    <div class="title" style="padding-bottom: 30px; padding-top: 0px">
		                      <div class="icon"><i class="icon-info"></i></div><strong style="color: white">Current °C</strong>
		                    </div>
		                    <div class="number dashtext-1" id="Temperature2" style="padding-bottom: 30px">
		                    	25 °C
		                    </div>
		                 </div>
		                </div>
		              </div>
		            </div>
		          </div>
		        </section>
		        
		        <div id=showView style="margin-top: 30px; margin-left: 30px">
             		<div style="background-color: #864DD9; width: 490px; color: white; font-weight: bold;justify-content: center;">Object Detection Situation</div>
             		<div style="background-color: transparent; width: 5px"></div>
             		<div id=batteryMode style="background-color: #864DD9; width: 380px; color: white; font-weight: bold;justify-content: center;">
             			<div style="width: 190px">
	              			<input id="modeOn2" onclick="manual2('On')" value="Manual Driving Mode" readonly="readonly" style="border-color: transparent; background-color: dimgray; text-align: center; color: white; width: 190px; font-weight: bold;justify-content: center;">
	              		</div>
						<div  style="width: 190px">
							<input id="modeOff2" onclick="manual2('Off')" value="Auto Driving Mode" readonly="readonly" style="border-color: transparent; background-color: #ADFF2F; text-align: center; color: black; width: 190px; font-weight: bold;justify-content: center;">
						</div>
             		</div>
             	</div>
             	
             	<!-- 주행 중 탐지한 이미지 보여주기 -->
             	<img id=driveView2 style="width: 325px; height: 275px; position: absolute; top: 617px; left: 63px"/>
             	<img id=detectView2 src="${pageContext.request.contextPath}/resource/img/milk.jpg" style="width: 140px; height: 140px; position: absolute; top: 750px; left: 250px; opacity: 0.8; display: none;"/>
             	
             	<!-- 주행 중 탐지하는 표지판 내용 표시 -->
             	<div class="driving-sign" style="height: 51px; top: 617px">Detected Object </div>
             		<input id="driveObject2" value="N/A" readonly="readonly" style="background-color: transparent; border-color: transparent; position: absolute; color: white; left: 390px; font-weight: bold; font-size:x-large; top:632px ">
             	
             	<div class="driving-sign" style="height: 51px; top: 672px">Detected Sign </div>
					<input id="driveSign2" value="N/A" readonly="readonly" style="background-color: transparent; border-color: transparent; position: absolute; color: white; left: 390px; font-weight: bold; font-size:x-large; top:687px ">
				
				<div class="driving-sign" style="height: 51px;top: 727px">Detected Zone </div>
					<input id="driveLoc2" value="N/A" readonly="readonly" style="background-color: transparent; border-color: transparent; position: absolute; color: white; left: 390px; font-weight: bold; font-size:x-large; top:742px ">
				
				<div class="driving-sign" style="height: 51px; top: 782px">Target Speed </div>
					<input id="driveSpeed2" value="N/A" readonly="readonly" style="background-color: transparent; border-color: transparent; position: absolute; color: white; left: 390px; font-weight: bold; font-size:x-large; top:797px ">
				
				<div class="driving-sign" style="height: 55px;top: 837px">Target Action </div>
					<input id="driveAction2" value="N/A" readonly="readonly" style="background-color: transparent; border-color: transparent; position: absolute; color: white; left: 390px; font-weight: bold; font-size:x-large; top:852px ">

				<div id="manual_control2" style="display:none; width: 380px; height: 280px">
					<div style="margin-left: 525px; width: 190px; height:280px; position: absolute" align="center">
						<!-- 키보드로 출발/정지 -->
						<input value="Motor Ctrl" style="background-color: transparent; border-color: transparent; font-weight: bold; font-size: large; color: white; text-align: center; width: 190px"></br>
						<a class="btn btn-outline-warning btn-lg" id="up2" onmousedown="tire_button_down('up')" onmouseup="tire_button_up('up')" onclick="click_up()"style=" margin-bottom:5px ;border-color:#ADFF2F; border-width: medium; font-weight: bold;">↑</a><br/>
						<a class="btn btn-outline-warning btn-lg" id="left2" onmousedown="tire_button_down('left')" onmouseup="tire_button_up('left')" onclick="click_left()" style="border-color:#ADFF2F; border-width: medium; font-weight: bold;">←</a>
						<a class="btn btn-outline-warning btn-lg" id="down2" onmousedown="tire_button_down('down')" onmouseup="tire_button_up('down')" onclick="click_down()" style="border-color:#ADFF2F; border-width: medium; font-weight: bold;">□</a>
						<a class="btn btn-outline-warning btn-lg" id="right2" onmousedown="tire_button_down('right')" onmouseup="tire_button_up('right')" onclick="click_right()" style="border-color:#ADFF2F; border-width: medium; font-weight: bold;">→</a></br>
					
						<a class="btn btn-outline-warning btn-sm manual-line" onclick="manual2('W')" style="color: black">L Line</a>
						<a class="btn btn-outline-warning btn-sm manual-line" onclick="manual2('W')" style="color: black">R Line</a>
					</div>
					
					<div style="margin-left: 715px; width: 190px; height:280px; position: absolute" align="center" >
						<!-- 센서 띄우기 -->
						<input value="Sensor Ctrl" style="background-color: transparent; border-color: transparent; font-weight: bold; font-size: large; color: white; text-align: center; width: 190px"></br>
						<a class="btn btn-outline-warning btn-sm manual-button" onclick="manual2('W')" style="border-color:darkred;  color: darkred">RED</a>
						<a class="btn btn-outline-warning btn-sm manual-button" onclick="manual2('S')" style="border-color:white;  color: white">SIREN</a>
						<a class="btn btn-outline-warning btn-sm manual-button" onclick="manual2('W')" style="border-color:gold;  padding-left: 8px; margin-top: 10px; color: gold">YELLOW</a>
						<a class="btn btn-outline-warning btn-sm manual-button" onclick="manual2('W')" style="border-color:white;  padding-left: 8px; margin-top: 10px; color: white">FLASH</a></br>
						<a class="btn btn-outline-warning btn-sm manual-button" onclick="manual2('W')" style="border-color:green; margin-top: 10px; color: green">GREEN</a>
						<a class="btn btn-outline-warning btn-sm manual-button" onclick="manual2('W')" style="border-color:white; margin-top: 10px; color: white">???</a>
					</div>
				</div>
	</body>
</html>