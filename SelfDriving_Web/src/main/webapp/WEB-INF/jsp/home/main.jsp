﻿<%@ page contentType="text/html; charset=UTF-8"%>
<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core"%>
<%@ taglib prefix="fmt" uri="http://java.sun.com/jsp/jstl/fmt"%>

<!DOCTYPE html>
<html>
  <head> 
	    <meta charset="utf-8">
		<meta http-equiv="X-UA-Compatible" content="IE=edge">
	    <title>AIOT FINAL PROJECT | TEAM 2</title>
	    <meta name="viewport" content="width=device-width, initial-scale=1">
	    <link rel="stylesheet" href="${pageContext.request.contextPath}/resource/jquery-ui/jquery-ui.min.css">
		<script src="${pageContext.request.contextPath}/resource/jquery-ui/jquery-ui.min.js"></script>
		<script src="https://cdnjs.cloudflare.com/ajax/libs/paho-mqtt/1.0.1/mqttws31.min.js" type="text/javascript"></script>
	    <!--  Template 관련 설정 파일들 -->
	    <!-- Bootstrap CSS-->
	    <link rel="stylesheet" href="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/bootstrap/css/bootstrap.min.css">
	    <!-- Font Awesome CSS-->
	    <link rel="stylesheet" href="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/font-awesome/css/font-awesome.min.css">
	    <!-- Custom Font Icons CSS-->
	    <link rel="stylesheet" href="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/css/font.css">
	    <!-- Google fonts - Muli-->
	    <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Muli:300,400,700">
	    <!-- theme stylesheet-->
	    <link rel="stylesheet" href="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/css/style.default.css" id="theme-stylesheet">
	    <!-- Custom stylesheet - for your changes-->
	    <link rel="stylesheet" href="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/css/custom.css">
	    <!-- Favicon-->
	    <link rel=icon href="${pageContext.request.contextPath}/resource/img/jetracer.png">

		<script src="${pageContext.request.contextPath}/resource/jquery/jquery-3.4.1.min.js" crossorigin="anonymous"></script>
		<script src="${pageContext.request.contextPath}/resource/popper/popper.min.js"></script>

		
		
		<link href="${pageContext.request.contextPath}/resource/bootstrap/css/change.css" rel="stylesheet">
		
		<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">

		 <style>
			#div1 {font-size:48px;}
			.no-gutters {
			  margin-right: 0;
			  margin-left: 0;
			  > .col,
			  > [class*="*cols-"] {
			    padding-right: 0;
			    padding-left: 0;
			  }
			} 
			#jetracer {
		  width: 100px;
		  height: 100px;
		  position: relative;
		  animation: myfirst 5s infinite;
		  animation-direction: normal;
		}

			@keyframes myfirst {
			  0%   {left: 0%; top: 0%;}
			  25%  {left: 100%; top: 0%;}
			  50%  {left: 100%; top: 100%;}
			  75%  {left: 0%; top: 100%;}
			  100% {left: 0%; top: 0%;}
		}
		body{
		font-size: 16px;
		}
</style>
		<script>
			$(function(){
				client = new Paho.MQTT.Client(location.hostname, 61614, new Date().getTime().toString());
				client.onMessageArrived = onMessageArrived;
				client.connect({onSuccess:onConnect});
			});
			
			function onConnect() {
				console.log("mqtt broker connected")
				client.subscribe("/1cctv");
				client.subscribe("/2cctv");
				client.subscribe("/3cctv");
				client.subscribe("/4cctv");
				client.subscribe("/sensor");
			}
			
			function onMessageArrived(message) {
 				if(message.destinationName =="/1cctv") {
 					const json = message.payloadString;
					const obj = JSON.parse(json);
					$("#cameraView1").attr("src", "data:image/jpg;base64,"+ message.payloadString);
				}
				if(message.destinationName =="/2cctv") {
					const json = message.payloadString;
					const obj = JSON.parse(json);
					$("#cameraView2").attr("src", "data:image/jpg;base64,"+ obj.Cam);
				}
				if(message.destinationName =="/3cctv") {
					const json = message.payloadString;
					const obj = JSON.parse(json);
					$("#cameraView3").attr("src", "data:image/jpg;base64,"+ message.payloadString);
				}
				if(message.destinationName =="/4cctv") {
					const json = message.payloadString;
					const obj = JSON.parse(json);
					$("#cameraView4").attr("src", "data:image/jpg;base64,"+ obj.Cam);
				}
			}
		</script>		 
	</head>
	
	<body>
		<header class="header">   
	      <nav class="navbar navbar-expand-lg">
	        <div class="container-fluid d-flex align-items-center justify-content-between">
	          <div class="navbar-header">
	            <a href="${pageContext.request.contextPath}/home/main.do" class="navbar-brand">
		              <div class="brand-text brand-big visible text-uppercase" style="font-size: x-large"><strong class="text-primary">AIOT</strong><strong>Admin</strong></div>
		              <div class="brand-text brand-sm"><strong class="text-primary">A</strong><strong>A</strong></div>
		         </a>
	            <!-- Sidebar Toggle Btn-->
	            <button class="sidebar-toggle"><i class="fa fa-long-arrow-left"></i></button>
	          </div>
	          <div class="right-menu list-inline no-margin-bottom">    
	            <!-- Languages dropdown    -->
	            <div class="list-inline-item dropdown"><a id="languages" rel="nofollow" data-target="#" href="#" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false" class="nav-link language dropdown-toggle"><img src="img/flags/16/GB.png" alt=""><span class="d-none d-sm-inline-block">LOGIN</span></a>
	              <div aria-labelledby="languages" class="dropdown-menu"><a rel="nofollow" href="#" class="dropdown-item"> <img src="img/flags/16/DE.png" alt="" class="mr-2"><span>German</span></a><a rel="nofollow" href="#" class="dropdown-item"> <img src="img/flags/16/FR.png" alt="English" class="mr-2"><span>French  </span></a></div>
	            </div>
	            <!-- Log out               -->
	            <div class="list-inline-item logout"><a id="logout" href="login.html" class="nav-link"> <span class="d-none d-sm-inline">Logout </span><i class="icon-logout"></i></a></div>
	          </div>
	        </div>
	      </nav>
	    </header>
	    
		<div class="d-flex align-items-stretch">
	      <!-- Sidebar Navigation-->
	      <nav id="sidebar">
	        <!-- Sidebar Header-->
	        <div class="sidebar-header d-flex align-items-center">
	          <div class="avatar" style="width: 100px; height: 100px; align-itself: center; "><img src="${pageContext.request.contextPath}/resource/img/milk.jpg" class="img-fluid rounded-circle"></div>
	          <div class="title">
	            <h1 class="h5" style="color: lightgray">AIoT Project</h1>
	            <p style="color: lightgray">Team 2</p>
	          </div>
	        </div>
	        <!-- Sidebar Navidation Menus--><span class="heading" style="color:lightgray ;">MENU</span>
	        <ul class="list-unstyled">
	          <li class="active"><a href="${pageContext.request.contextPath}/home/main.do" style="color: lightgray"> <i class="icon-home"></i>MAIN DASHBOARD </a></li>
	          <li><a href="${pageContext.request.contextPath}/home/jetracer.do" style="color: lightgray"> <i class="icon-writing-whiteboard"></i>JET-RACERS </a></li>
	          <li><a href="${pageContext.request.contextPath}/home/history.do" style="color: lightgray"> <i class="icon-grid"></i>HISTORY </a></li>
	          <li><a href="${pageContext.request.contextPath}/home/status.do" style="color: lightgray"> <i class="icon-padnote"></i>REAL-TIME STATUS </a></li>
	          <li><a href="${pageContext.request.contextPath}/home/analysis.do" style="color: lightgray"> <i class="icon-chart"></i>ANALYSIS </a></li>
	      </nav>
	      
	      <div class="page-content">

	      <section class="no-padding-top">
	          <div class="container-fluid">
	        	<div class="container" style="position:relative; margin-right: 0px; margin-left:0px; width: 100%; height:auto; margin-top: 20px;border: 1px solid gold">
	            <div class="row row-cols-3">	            
	                  <div class="col" style="padding-left: 0px; padding-right: 0px; width: 50%; height: 50%;border: 1px solid gold">
	                  	<img id=faceimg src= "${pageContext.request.contextPath}/resource/img/face_good.png"style="width: 100%; height:auto;padding-left: 0px; padding-right: 0px;"/>
	                  	</div>
	                  	<div class="col" style="padding-left: 0px; padding-right: 0px; width: 50%; height: 50%;border: 1px solid gold">
						<p style="font-size: 3rem">안전합니다 여러분!</p>
						</div>
	                  </div>
	                   
	             </div>
	             </div>
	       </section>
 	     <section style="padding-right: 0px">
          <div class="container-fluid">
         	<div class="container" style="position:absolute; margin-right: 0px; margin-left: 0px; width: 600px; height: 600px; margin-top: 20px;">
			  <div class="row row-cols-3" style="background-image:url('${pageContext.request.contextPath}/resource/img/track.png');background-repeat : no-repeat;  background-size:contain ">
			    <div class="col" style="padding-left: 0px; padding-right: 0px; width: 200px; height: 200px"><img id=position1 src= "${pageContext.request.contextPath}/resource/img/jetracer.png"style="width: 300px; height: 300px; padding-left: 0px; padding-right: 0px;display: none;"/></div>
			    <div class="col" style="padding-left: 0px; padding-right: 0px; width: 200px; height: 200px"><img id=position2 src= "${pageContext.request.contextPath}/resource/img/jetracer.png"style="width: 300px; height: 300px; padding-left: 0px; padding-right: 0px;display: none;"/></div>
			    <div class="col" style="padding-left: 0px; padding-right: 0px; width: 200px; height: 200px"><img id=position3 src= "${pageContext.request.contextPath}/resource/img/jetracer.png"style="width: 300px; height: 300px; padding-left: 0px; padding-right: 0px;display: none;"/></div>
			    
			    <div class="col" style="padding-left: 0px; padding-right: 0px; width: 200px; height: 200px"><img id=position8 src= "${pageContext.request.contextPath}/resource/img/jetracer.png"style="width: 300px; height: 300px; padding-left: 0px; padding-right: 0px;display: none;"/></div>
			    <div class="col" style="padding-left: 0px; padding-right: 0px; width: 200px; height: 200px"><img id=position9 src= "${pageContext.request.contextPath}/resource/img/jetracer.png"style="width: 300px; height: 300px; padding-left: 0px; padding-right: 0px;display: none;"/></div>
			    <div class="col" style="padding-left: 0px; padding-right: 0px; width: 200px; height: 200px"><img id=position4 src= "${pageContext.request.contextPath}/resource/img/jetracer.png"style="width: 300px; height: 300px; padding-left: 0px; padding-right: 0px;display: none;"/></div>
			    
			    <div class="col" style="padding-left: 0px; padding-right: 0px; width: 200px; height: 200px"><img id=position7 src= "${pageContext.request.contextPath}/resource/img/jetracer.png"style="width: 300px; height: 300px; padding-left: 0px; padding-right: 0px;display: none;"/></div>
			    <div class="col" style="padding-left: 0px; padding-right: 0px; width: 200px; height: 200px"><img id=position6 src= "${pageContext.request.contextPath}/resource/img/jetracer.png"style="width: 300px; height: 300px; padding-left: 0px; padding-right: 0px;display: none;"/></div>
			    <div class="col" style="padding-left: 0px; padding-right: 0px; width: 200px; height: 200px"><img id=position5 src= "${pageContext.request.contextPath}/resource/img/jetracer.png"style="width: 300px; height: 300px; padding-left: 0px; padding-right: 0px;display: none;"/></div>
			   </div>
			</div>
          </div>
        </section>
        
        <section style="padding-right: 0px">
          <div class="container-fluid">
         	<div class="container" style="position:absolute; margin-right: 0px; margin-left: 800px; width: 600px; height: 600px; margin-top: 20px;">
			  <div class="row row-cols-2" style="background-image:url('${pageContext.request.contextPath}/resource/img/track.png');background-repeat : no-repeat;  background-size:contain ">
			  <img id=jetracer src= "${pageContext.request.contextPath}/resource/img/jetracer.png"style="width: 300px; height: 300px; padding-left: 0px; padding-right: 0px;"/>
			   </div>
			  
			</div>
          </div>
        </section>
        <script>
        count = 0;
        x = 0;
        count2 = 0;
        x2 = 0;
        x3 = 0;
        $(document).ready(function() {
		    setInterval(move,100)
		    setInterval(move2,100)
        });
        function move() {
        	$("#position"+x).hide();
        	count++;
        	x= count%8+1;
			$("#position"+x).show();
		}
        function move2() {
        	$("#position2"+x3).hide();
        	count2++;
        	x2= count2%8+1;
        	x3= 9-x2
			$("#position2"+x3).show();
		}
        </script>
        <button onclick="move()">눌러요.</button>
        
        <!-- JavaScript files-->
	   	<!--  <script src="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/jquery/jquery.min.js"></script> -->
	    <script src="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/popper.js/umd/popper.min.js"> </script>
	    <script src="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/bootstrap/js/bootstrap.min.js"></script>
	    <script src="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/jquery.cookie/jquery.cookie.js"> </script>
	    <script src="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/jquery-validation/jquery.validate.min.js"></script>
<!-- 	<script src="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/js/charts-home.js"></script> -->
	    <script src="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/js/front.js"></script>
<!--    <script src="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/chart.js/Chart.min.js"></script> -->
</body>
</html>