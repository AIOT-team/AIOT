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
		  		<div style="background-color: dimgray; height: 900px; width: 1200px; position: absolute; margin-left:22px; margin-top: 22px; text-align: center; font-weight: bolder; font-size: 300px;">여기에 HUD</div>
  				
  				<input value="Battery Charging Status" readonly="readonly" style="border-color: transparent ; background-color: #864DD9 ; text-align: center; color: white; font-weight: bold;justify-content: center;width: 320px; position: absolute; top: 150px; left: 1260px; height: 30px;">
  				<div style="position: absolute; top: 180px; left: 1260px; height: 30px; display: flex; flex-direction:  row;">
            		<div>
            			<input value="Battery" readonly="readonly" style="border-color: transparent; background-color: #ADFF2F; text-align: center; color: black;  font-weight: bold;justify-content: center; width: 160px; height: 30px">
            		</div>
					<div>
						<input value="Adapter Connected" readonly="readonly" style="border-color: transparent; background-color:dimgray; text-align: center; color: white;  font-weight: bold;justify-content: center;width: 160px; height: 30px">
					</div>
	           	</div>
				
				<div class="statistic-block block" style="width: 320px; height: 100px; margin-bottom: 10px; padding-bottom: 0px; margin-left: 1232px; position: absolute; top:210px">
                  <div class="progress-details d-flex align-items-end justify-content-between">
                    <div class="title">
                      <div class="icon"><i class="icon-writing-whiteboard"></i></div><strong style="color: white">Battery Status</strong>
                    </div> 
                    <div class="number dashtext-1" id="jetRacerText1">
                    	100%
                    </div>
                  </div>
                  <div class="progress progress-template">
                    <div id="jet1Battery" role="progressbar" style="width: 100%" aria-valuenow="30" aria-valuemin="0" aria-valuemax="100" class="progress-bar progress-bar-template dashbg-1"></div>
                  </div>
                </div>
				
  				<input value="Motion Detection" readonly="readonly" style="border-color: transparent ; background-color: #864DD9 ; text-align: center; color: white;  font-weight: bold;justify-content: center;width: 320px; position: absolute; top: 325px; left: 1260px; height: 30px;">
  				<img id=motionView style="width: 320px; height: 280px; position: absolute; top: 355px; left: 1260px"/>
				
				<section class="no-padding-top no-padding-bottom" style="top:640px; left: 1260px; position: absolute;padding: 0px">
		          <div class="container-fluid">
		            <div class="row" style="width: 320px; height: 200px" >
		              <div class="col-md-3 col-sm-6" style="padding: 0px; height: 170px">
		                <div class="statistic-block block" style="justify-content: center; padding: 0px; width: 320px; margin-bottom: 10px">
		                  <div class="progress-details d-flex align-items-end justify-content-between" style="justify-content: center;height: 80px; padding: 0px">
		                    <div class="title" style="justify-content: center; margin: 15px">
		                      <div class="icon"><i class="icon-info"></i></div><strong style="color: white">Detected Motion</strong>
		                    </div>
		                    <div id="motionName1" style="color: #864DD9; font-size: x-large;  font-weight:bold; ; margin-right: 15px; margin-bottom: 15px">
		                    	---
		                    </div>
		                 </div>
		                </div>
 		                <div class="statistic-block block" style="justify-content: center; padding: 0px; width: 320px">
		                  <div class="progress-details d-flex align-items-end justify-content-between" style="justify-content: center; height: 80px">
		                    <div class="title" style="justify-content: center; margin: 15px">
		                      <div class="icon"><i class="icon-info"></i></div><strong style="color: white">Target Action</strong>
		                    </div>
		                    <div id="targetSpeed1" style="color: #864DD9; font-size: x-large;  font-weight:bold; ; margin-right: 15px; margin-bottom: 15px">
		                    	Speed : ---
		                    </div>
		                 </div>
		                </div>
		              </div>
		            </div>
		          </div>
		        </section>

		        <input value="Driving Mode" readonly="readonly" style="border-color: transparent ; background-color: #864DD9 ; text-align: center; color: white;font-weight: bold;justify-content: center; width: 320px; position: absolute; top: 820px; left: 1260px; height: 30px">
		        <div id=batteryMode style="background-color: #864DD9; width:320px; color: white; font-weight: bold;justify-content: center; position: absolute; left: 1260px; top:820px">
           			<div style="width:160px">
             			<input id="modeOn" onclick="manual('On')" value="Manual Driving" readonly="readonly" style="border-color: transparent; width: 160px; background-color: dimgray; text-align: center; color: white; font-weight: bold;justify-content: center;">
             		</div>
					<div  style="width: 160px">
						<input id="modeOff" onclick="manual('Off')" value="Auto Driving" readonly="readonly" style="border-color: transparent; width: 160px; background-color: #ADFF2F; text-align: center; color: black; font-weight: bold;justify-content: center;">
					</div>
           		</div>
				
				<div id="manual_control" style="display:none ; width: 380px; height: 200px; position: absolute; top: 840px;">
					<div style="margin-left: 1170px; width: 320px; height:200px; position: absolute" align="center">
						<input value="Motor Ctrl" style="background-color: transparent; border-color: transparent; font-weight: bold; font-size: large; color: white; text-align: center; width: 160px; display: none;"></br>
						<a class="btn btn-outline-warning btn-lg" id="up" onmousedown="tire_button_down('up')" onmouseup="tire_button_up('up')" onclick="click_up()"style=" margin-bottom:5px ;border-color:#ADFF2F; border-width: medium; font-weight: bold;">↑</a><br/>
						<a class="btn btn-outline-warning btn-lg" id="left" onmousedown="tire_button_down('left')" onmouseup="tire_button_up('left')" onclick="click_left()" style="border-color:#ADFF2F; border-width: medium; font-weight: bold;">←</a>
						<a class="btn btn-outline-warning btn-lg" id="down" onmousedown="tire_button_down('down')" onmouseup="tire_button_up('down')" onclick="click_down()" style="border-color:#ADFF2F; border-width: medium; font-weight: bold;">□</a>
						<a class="btn btn-outline-warning btn-lg" id="right" onmousedown="tire_button_down('right')" onmouseup="tire_button_up('right')" onclick="click_right()" style="border-color:#ADFF2F; border-width: medium; font-weight: bold;">→</a></br>
 						<a class="btn btn-outline-warning btn-sm manual-line" onclick="manual('W')" style="color: black">L Line</a>
						<a class="btn btn-outline-warning btn-sm manual-line" onclick="manual('W')" style="color: black">R Line</a>
					</div>
			
					<div style="margin-left: 1350px; width: 190px; height:200px; position: absolute" align="center">
						<input ="Sensor Ctrl" readonly="readonly" style="background-color: transparent; border-color: transparent; font-weight: bold; font-size: large; color: white; text-align: center; width: 190px"></br>
<!-- 					<a class="btn btn-outline-warning btn-sm manual-button" onclick="manual('W')" style="border-color:darkred;  color: darkred">RED</a> 
						<a class="btn btn-outline-warning btn-sm manual-button" onclick="manual('S')" style="border-color:white;  color: white">SIREN</a>
 						<a class="btn btn-outline-warning btn-sm manual-button" onclick="manual('W')" style="border-color:gold;  padding-left: 8px; margin-top: 10px; color: gold">YELLOW</a>
						<a class="btn btn-outline-warning btn-sm manual-button" onclick="manual('W')" style="border-color:white;  padding-left: 8px; margin-top: 10px; color: white">FLASH</a></br>
						<a class="btn btn-outline-warning btn-sm manual-button" onclick="manual('W')" style="border-color:green; margin-top: 10px; color: green">GREEN</a> -->
					</div>
				</div>
	</body>
</html>