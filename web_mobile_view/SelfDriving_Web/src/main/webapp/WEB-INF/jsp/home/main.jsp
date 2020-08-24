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

	    <link rel="stylesheet" href="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/bootstrap/css/bootstrap.min.css">
	    <link rel="stylesheet" href="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/font-awesome/css/font-awesome.min.css">
	    <link rel="stylesheet" href="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/css/font.css">
	    <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Muli:300,400,700">
	    <link rel="stylesheet" href="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/css/style.default.css" id="theme-stylesheet">
	    <link rel=icon href="${pageContext.request.contextPath}/resource/img/jetracer.png">

		<script src="${pageContext.request.contextPath}/resource/jquery/jquery-3.4.1.min.js" crossorigin="anonymous"></script>
		<script src="${pageContext.request.contextPath}/resource/popper/popper.min.js"></script>

		<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">

	    <script src="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/popper.js/umd/popper.min.js"> </script>
	    <script src="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/bootstrap/js/bootstrap.min.js"></script>
	    <script src="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/jquery.cookie/jquery.cookie.js"> </script>
	    <script src="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/jquery-validation/jquery.validate.min.js"></script>
	    <script src="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/js/front.js"></script>
		
		<link rel="stylesheet" href="${pageContext.request.contextPath}/resource/css/yunjis.css">
		<style>
		  .carousel-inner img {
		    width: 100%;
		    height: 400px;
		  }
		 </style>
	</head>
	<script>
	let ipid;
	var lastSendtimearr = [Date.now(), Date.now(), Date.now(),Date.now(),Date.now(),Date.now(),Date.now()];
	var subList=["1jetracer", "2jetracer","3jetracer","1cctv","2cctv","3cctv","4cctv"];
	
	$(function(){
		ipid = new Date().getTime().toString()
		client = new Paho.MQTT.Client("192.168.3.105", 61614, ipid);
		client.onMessageArrived = onMessageArrived;
		client.connect({onSuccess:onConnect});
	});
	
	function onConnect() {
		console.log("mqtt broker connected")

		client.subscribe("/mirror");
		client.subscribe("/req/1cctv");
		client.subscribe("/req/2cctv");
		client.subscribe("/req/3cctv");
		client.subscribe("/req/4cctv");
		client.subscribe("/req/1jetracer");
		client.subscribe("/req/2jetracer");
		client.subscribe("/req/3jetracer");
	}
	
	$(document).ready(function() {
	    setInterval(getinterval, 750);
	});  
	 
	 var lastSendtime=Date.now();
	 
	 function getinterval(){
		nowtime= Date.now();
		lastSendtimearr.forEach(function(element, index, array){
			interval=nowtime-element
			if(interval>750){
				console.log("연결이 끊긴다음"+subList[index]+ "몇초가 흘렀는지를 보여주는 console.log의 시간:"+interval);
				response(index);
			}
		});
	}
	 
 	function response(index){
		//console.log(subList[index]+"에게 답장을 보내쥬");
		message = new Paho.MQTT.Message(ipid);
		message.destinationName = "/res/"+subList[index];
		client.send(message);
	}
	
	function onMessageArrived(message) {
		if(message.destinationName =="/mirror") {
			const json = message.payloadString;
			const obj = JSON.parse(json);
			if(obj.direction=="left"){
				alert("최상단 페이지입니다.")
			}
			else if (obj.direction=="right"){
				location.href="jetracer.do";
			}
		}
		if(message.destinationName =="/req/1jetracer") {
			response(0);
			lastSendtimearr[0] = Date.now();
			$("#1jetracer").text("CONNECTED");
			$("#1jetracer").css("color", "#ADFF2F");
		}
		if(message.destinationName =="/req/2jetracer") {
			response(1);
			lastSendtimearr[1] = Date.now();
			$("#2jetracer").text("CONNECTED");
			$("#2jetracer").css("color", "#ADFF2F");
		}
		if(message.destinationName =="/req/3jetracer") {
			response(2);
			lastSendtimearr[2] = Date.now();
			$("#3jetracer").text("CONNECTED");
			$("#3jetracer").css("color", "#ADFF2F");
		}
		if(message.destinationName =="/req/1cctv") {
			response(3);
			lastSendtimearr[3] = Date.now();
			$("#1cctv").text("CONNECTED");
			$("#1cctv").css("color", "#ADFF2F");
		}
		if(message.destinationName =="/req/2cctv") {
			response(4);
			lastSendtimearr[4] = Date.now();
			$("#2cctv").text("CONNECTED");
			$("#2cctv").css("color", "#ADFF2F");
		}
		if(message.destinationName =="/req/3cctv") {
			response(5);
			lastSendtimearr[5] = Date.now();
			$("#3cctv").text("CONNECTED");
			$("#3cctv").css("color", "#ADFF2F");
		}
		if(message.destinationName =="/req/4cctv") {
			response(6);
			lastSendtimearr[6] = Date.now();
			$("#4cctv").text("CONNECTED");
			$("#4cctv").css("color", "#ADFF2F");
		}
	}

	</script>
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
	          <div class="avatar" style="width: 100px; height: 100px; align-itself: center; "><img id="adminFace" src="${pageContext.request.contextPath}/resource/img/milk.jpg" class="img-fluid rounded-circle"></div>
	          <div class="title">
	            <h1 class="h5" style="color: lightgray">AIoT Project</h1>
	            <p style="color: lightgray">관리자</p>
	          </div>
	        </div>
	        <span class="heading" style="color:#DB6574">CATEGORIES</span>
	        <ul class="list-unstyled">
	          <li class="active"><a href="${pageContext.request.contextPath}/home/main.do" style="color: lightgray"> <i class="icon-home"></i>메인 페이지 </a></li>
	          <li><a href="${pageContext.request.contextPath}/home/jetracer.do" style="color: lightgray"> <i class="icon-writing-whiteboard"></i>탐지봇 현황 </a></li>
	          <li><a href="${pageContext.request.contextPath}/home/history.do" style="color: lightgray"> <i class="icon-grid"></i>탐지 히스토리 조회 </a></li>
	          <li><a href="${pageContext.request.contextPath}/home/status.do" style="color: lightgray"> <i class="icon-padnote"></i>실시간 탐지 | 대응 현황 </a></li>
	          <li><a href="${pageContext.request.contextPath}/home/analysis.do" style="color: lightgray"> <i class="icon-chart"></i>탐지 결과 분석 </a></li>
	        </ul>
	      </nav>
	     
	       <div class="page-content" style="top: -50px; height: 1080px; padding-bottom: 0px;">
	       	
	       	<div style="margin-bottom: 10px; margin-top: 80px; color:white; font-weight: bold; font-size: xx-large; height: 50px; width: 1623px; text-align: center">지능형 농가 유해동물 탐지 및 대응 시스템</div>
	       	
	       	<div style="width: 810px; position: absolute; left: 50px; top: 150px; height: 436px">
				<input value="유해동물 탐지 기준 및 상태 요약" readonly="readonly" style="color:#FFFFE0 ; background-color: transparent ;border-color: #ADFF2F; border-style:solid;border-left:none; border-right:none; border-top:none ;border-width:medium; font-weight: 500; font-size:20px; font-weight: bold; width: 810px; text-align: center; margin-left: -3px; height: 36px; margin-top: -3px"/>
				
				<div class="container" style="width: 810px;height: 190px ;position: absolute; left: -3px; background-color: #2d3035 ; top:40px; padding: 0px">
		         <div class="table-responsive">
     			  <div style=" text-align: center; justify-content: center">
                   <table class="table table-sm" style="color: white; margin-top: 10px">
                     <thead>
                       <tr style="border-bottom: medium; border-style: double; border-color: white; border-left: none; border-right: none ; border-top: none;">
                         <th style="width: 100px">등급</th>
                         <th style="width: 300px">등급 분류 기준</th> 
                         <th style="width: 400px">유해 동물 리스트</th> 
                       </tr>
                     </thead>
                     <tbody style="font-size:medium;">
                      	<tr>
                      	  <td style="color: red; font-weight :bold;font-size: large; ">A</td>
                      	  <td>인명 피해가 예상되는 경우</td>
                          <td>곰, 표범, 멧돼지, 늑대</td>
                        </tr>
                        <tr>
                      	  <td style="color: orange; font-weight :bold;font-size: large; ">B</td>
                          <td>가축 피해가 예상되는 경우</td>
                          <td>매, 여우, 너구리</td>
                        </tr>
                        <tr>
                      	  <td style="color: yellow; font-weight :bold;font-size: large; ">C</td>
                          <td>농작물 피해가 예상되는 경우</td>
                          <td>사슴, 까마귀, 토끼 </td>
                        </tr>
                        <tr>
                      	  <td style="color: #9ACD32; font-weight :bold;font-size: large; ">D</td>
                          <td>가축이거나 큰 해를 끼치지 않는 경우</td>
                          <td>닭, 소, 오리, 말, 돼지, 양, 고양이, 강아지</td>
                        </tr>
                    </tbody>
                   </table>
                  </div>
                </div>
	          </div>

              <div class="container" style="width: 500px;height: 197px ;position: absolute; left: -3px; background-color: #2d3035 ; top:240px; padding: 0px">
            	<input value="등급별 유해동물 누적 탐지 수" readonly="readonly" style="color: #F5DEB3; font-size:20px ; font-weight: bold;  background-color: transparent; border-color: transparent; text-align: center; width: 500px ;position: absolute;">
            	<input value="A" readonly="readonly" style="color: red; font-size:100px ; font-weight: bolder; border-style: double; background-color: transparent; border-color: transparent; text-align: center; width: 150px; height:126px; margin-left: 25px; margin-top: 25px">
            	<input value="B" readonly="readonly" style="color: orange; font-size:100px ; font-weight: bolder; border-style: double; background-color: transparent; border-color: transparent; text-align: center; width: 150px; height:126px; margin-top: 25px">
            	<input value="C" readonly="readonly" style="color: yellow; font-size:100px ; font-weight: bolder; border-style: double; background-color: transparent; border-color: transparent; text-align: center; width: 150px; height:126px; margin-top: 25px">
	        	<input id="Alev" name="tempA" value="10 건" readonly="readonly" style="color:white; background-color: transparent; border-color: transparent; font-size: x-large; width: 150px; text-align: center; margin-left: 25px" >
	        	<input id="Blev" name="tempB" value="20 건" readonly="readonly" style="color:white; background-color: transparent; border-color: transparent; font-size: x-large; width: 150px; text-align: center; " >
	        	<input id="Clev" name="tempC" value="30 건" readonly="readonly" style="color:white; background-color: transparent; border-color: transparent; font-size: x-large; width: 150px; text-align: center; " >
              </div>
            
              <div class="container" style="width: 300px; position: absolute; left: 506px; top: 240px; height: 197px; padding:0px; background-color: #2d3035">            
               	<div style="width: 300px; height: 197px; background-color: transparent">
               		<input value="최근 1시간 기준 유해 등급" readonly="readonly" style="color: #F5DEB3; font-size:20px ; font-weight: bold;  background-color: transparent; border-color: transparent; text-align: center; width: 290px ;position: absolute;">
               		<img id="faceimg" src= "${pageContext.request.contextPath}/resource/img/face_good.png" style="width: 120px; margin-top: 46px; margin-left: 30px;  "/>
               		<p id="notice" style="font-size:xx-large; font-weight: bold; text-align: center; position: absolute; left: 160px; top:110px">데이터베이스에 연결하지 못했습니다.</p>
               		<p id="daynotice" style="font-size:medium ; margin: 0px; text-align: center; position: absolute; left: 30px; top: 165px; color: white"></p>
               	</div>
              </div>
			</div>
			
			<div style="width: 810px; position: absolute; left: 50px; top: 630px; height: 401px">
				<input value="자율주행 모드 처리 기준" readonly="readonly" style="color:#FFFFE0 ;background-color: transparent; border-color: #ADFF2F; border-style:solid;border-left:none; border-right:none; border-top:none ;border-width:medium; font-weight: 500; font-size:20px; font-weight: bold; width: 810px; text-align: center; margin-left: -3px; height: 36px; margin-top: -3px"/>

		        <div class="container" style="width: 810px; height: 365px; ;position: absolute; left: -3px; background-color: #2d3035 ; top:36px; padding: 0px; ">
		         <div class="table-responsive">
     			  <div style=" text-align: center; justify-content: center">
                   <table class="table table-sm" style="color: white; margin-top: 10px">
                     <thead>
                       <tr style="border-bottom: medium; border-style: double; border-color: white; border-left: none; border-right: none ; border-top: none;">
                         <th style="width: 150px">교통 사인 종류</th>
                         <th style="width: 200px">탐지 내용</th> 
                         <th style="width: 460px">타겟 주행 모드</th> 
                       </tr>
                     </thead>
                     <tbody style="font-size:medium;">
                      	<tr>
                      	  <td>여기에 뭐쓰지???...</td>
                      	  <td>모하게</td>
                          <td>디디디디디디디디디디디디</td>
                        </tr>
                        <tr>
                      	  <td>???</td>
                      	  <td>모하게</td>
                          <td>디디디디디디디디디디디디</td>
                        </tr>
                        <tr>
                      	  <td>???</td>
                      	  <td>모하게</td>
                          <td>디디디디디디디디디디디디</td>
                        </tr>
                        <tr>
                      	  <td>???</td>
                      	  <td>모하게</td>
                          <td>디디디디디디디디디디디디</td>
                        </tr>
                    </tbody>
                   </table>
                 </div>
               </div>
	         </div>
			</div>

			<div id="demo" class="carousel slide" data-ride="carousel" style="width:700px; position: absolute; left: 890px; top: 150px">
			  <input value="탐지봇 제원" readonly="readonly" style="background-color: transparent ; color: #FFFFE0 ; font-weight: 500; font-size:20px;border-color: #ADFF2F; border-width:medium; border-style:solid; border-left:none; border-right:none; border-top:none; font-weight: bold; width: 700px; text-align: center; margin-top: -3px"/>
			  <ul class="carousel-indicators">
			    <li data-target="#demo" data-slide-to="0" class="active"></li>
			    <li data-target="#demo" data-slide-to="1"></li>
			    <li data-target="#demo" data-slide-to="2"></li>
			  </ul>

			  <div class="carousel-inner" style="margin-top: 3px">
			    <div class="carousel-item active">
			      <img src="${pageContext.request.contextPath}/resource/img/farm.jpg">
			    </div>
			    <div class="carousel-item">
			      <img src="${pageContext.request.contextPath}/resource/img/farm2.jpg">
			    </div>
			    <div class="carousel-item">
			      <img src="${pageContext.request.contextPath}/resource/img/jet1.jpg">
			    </div>
			  </div>
			  
			  <a class="carousel-control-prev" href="#demo" data-slide="prev">
			    <span class="carousel-control-prev-icon"></span>
			  </a>
			  <a class="carousel-control-next" href="#demo" data-slide="next">
			    <span class="carousel-control-next-icon"></span>
			  </a>
			</div>
			
			<div style="width: 700px; position: absolute; left: 890px; top: 630px">
				<input value="탐지봇 및 CCTV 연결 상태" readonly="readonly" style="background-color: transparent ; color: #FFFFE0; font-weight: 500; font-size:20px; font-weight: bold; width: 700px; text-align: center;border-color: #ADFF2F; border-width:medium; border-style:solid; border-left:none; border-right:none; border-top:none;"/>
				
				<div class="container" style="width: 700px; height: 365px; ;position: absolute; left:0px; background-color: #2d3035 ; top:36px; padding: 0px; ">
		         <div class="table-responsive">
     			  <div style=" text-align: center;">
                   <table class="table table-sm" style="color: white; margin-top: 10px;">
                     <thead>
                       <tr style="border-bottom: medium; border-style: double; border-color: white; border-left: none; border-right: none ; border-top: none;">
                         <th style="width: 150px">연결 대상</th>
                         <th style="width: 200px">연결 상태</th> 
                         <th style="width: 450px">특이 사항</th> 
                       </tr>
                     </thead>
                     <tbody style="font-size: medium;">
                      	<tr style="height: 45px">
                      	  <td>Jet-Racer #1</td>
                      	  <td id="1jet">DISCONNECTED</td>
                          <td>디디디디디디디디디디디디</td>
                        </tr>
                       <tr style="height: 45px;">
                      	  <td>Jet-Racer #2</td>
                      	  <td id="2jet">DISCONNECTED</td>
                          <td>디디디디디디디디디디디디</td>
                        </tr>
                        <tr style="height: 45px;">
                      	  <td>Jet-Racer #3</td>
                      	  <td id="3jet">DISCONNECTED</td>
                          <td>디디디디디디디디디디디디</td>
                        </tr>
                        <tr style="height: 45px;">
                      	  <td>CCTV #1</td>
                      	  <td id="1cctv">DISCONNECTED</td>
                          <td>디디디디디디디디디디디디</td>
                        </tr>
                        <tr style="height: 45px;">
                      	  <td>CCTV #2</td>
                      	  <td id="2cctv">DISCONNECTED</td>
                          <td>디디디디디디디디디디디디</td>
                        </tr>
                        <tr style="height: 45px;">
                      	  <td>CCTV #3</td>
                      	  <td id="3cctv">DISCONNECTED</td>
                          <td>디디디디디디디디디디디디</td>
                        </tr>
                        <tr style="height: 45px;">
                      	  <td>CCTV #4</td>
                      	  <td id="4cctv">DISCONNECTED</td>
                          <td>디디디디디디디디디디디디</td>
                        </tr>
                    </tbody>
                   </table>
                 </div>
               </div>
	         </div>
			
			</div>
			
        </div>
     	</div>
     	
        <script>
        $.ajax({
			type: "POST",
			async: false,
			url: "${pageContext.request.contextPath}/home/mainDangerLevel.do",
			success:
   			function(howdanger){
					//DB에서 지금부터 1시간 내에 가장 위험한 등급을 반환한다.
					var d =new Date()
					if (howdanger == "" || howdanger == "D"){
						$("#faceimg").attr("src","${pageContext.request.contextPath}/resource/img/face_good.png")
						$("#notice").css("color", "#9ACD32");
						$("#notice").html("D-안전");
						$("#daynotice").html("기준 시각: " + d.getFullYear()+"년"+(d.getMonth()+1)+"월"+d.getDate()+"일"+d.getHours()+"시");
					}
					else if (howdanger == "C"){
						$("#faceimg").attr("src","${pageContext.request.contextPath}/resource/img/face_good.png")
						$("#notice").css("color", "yellow");
						$("#notice").html("C-주의");
						$("#daynotice").html("기준 시각: " + d.getFullYear()+"년"+(d.getMonth()+1)+"월"+d.getDate()+"일"+d.getHours()+"시");
					}
					else if (howdanger == "B"){
						$("#faceimg").attr("src","${pageContext.request.contextPath}/resource/img/face_soso.png")
						$("#notice").css("color", "orange");
						$("#notice").html("B-위험");
						$("#daynotice").html("기준 시각: " + d.getFullYear()+"년"+(d.getMonth()+1)+"월"+d.getDate()+"일"+d.getHours()+"시");
					}
					else if (howdanger == "A"){
						$("#faceimg").attr("src","${pageContext.request.contextPath}/resource/img/face_bad.png")
						$("#notice").css("color", "red");
						$("#notice").html("A-매우위험");
						$("#daynotice").html("기준 시각: " + d.getFullYear()+"년"+(d.getMonth()+1)+"월"+d.getDate()+"일"+d.getHours()+"시");
					}
			}
  		});
        
       	$.ajax({
   			type: "POST",
   			async: false,
   			url: "${pageContext.request.contextPath}/home/LevelCount.do",
   			success: 
   			function(levelCount){
  				console.log(typeof(levelCount));
   				console.log(levelCount);
   				
   				console.log("A등급: " + levelCount[0].dlevelCount);
   				console.log("B등급: " + levelCount[1].dlevelCount);
   				console.log("C등급: " + levelCount[2].dlevelCount);
   				console.log("D등급: " + levelCount[3].dlevelCount);
   				
   				//number구만
   				console.log(typeof(levelCount[0].dlevelCount));

   				var alev = $("input[name=tempA]").val(levelCount[0].dlevelCount +" 건");
   				var blev = $("input[name=tempB]").val(levelCount[1].dlevelCount +" 건");
   				var clev = $("input[name=tempC]").val(levelCount[2].dlevelCount +" 건");
   			}
   		});
        
        </script>

</body>
</html>