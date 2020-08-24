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
		<link rel="stylesheet" href="${pageContext.request.contextPath}/resource/jquery-ui/jquery-ui.min.css">
		<script src="${pageContext.request.contextPath}/resource/jquery-ui/jquery-ui.min.js"></script>
		<script src="https://cdnjs.cloudflare.com/ajax/libs/paho-mqtt/1.0.1/mqttws31.min.js" type="text/javascript"></script>

	    <link rel="stylesheet" href="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/bootstrap/css/bootstrap.min.css">
	    <link rel="stylesheet" href="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/font-awesome/css/font-awesome.min.css">
	    <link rel="stylesheet" href="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/css/font.css">
	    <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Muli:300,400,700">
	    <link rel="stylesheet" href="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/css/style.default.css" id="theme-stylesheet">
	    <link rel=icon href="${pageContext.request.contextPath}/resource/img/jetracer.png">
		
		<script src="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/bootstrap/js/bootstrap.min.js"></script>

		<script src="${pageContext.request.contextPath}/resource/popper/popper.min.js"></script>
		<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
		
	    <script src="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/popper.js/umd/popper.min.js"> </script>
	    <script src="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/bootstrap/js/bootstrap.min.js"></script>
	    <script src="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/jquery.cookie/jquery.cookie.js"> </script>
	    <script src="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/jquery-validation/jquery.validate.min.js"></script>
	    <script src="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/js/front.js"></script>
	    
	    <link rel="stylesheet" href="${pageContext.request.contextPath}/resource/css/yunjis.css">

		<script>
		var jetracer1connectedflag= false;
		var jetracer2connectedflag= false;
		var jetracer3connectedflag= false;
		
		var cctv1DetectingFlag = false;
		var cctv2DetectingFlag = false;
		var cctv3DetectingFlag = false;
		var cctv4DetectingFlag = false;
		
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
				
				RowClick();
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
						//console.log("연결이 끊긴다음"+subList[index]+ "몇초가 흘렀는지를 보여주는 console.log의 시간:"+interval);
						response(index);
					}
				});
			}

			function animalTable(){
				var currentLocation = window.location;
				$("#animalTable").load(currentLocation + ' #animalTable');
				console.log("table reloaded~~~!!!");
			}
			
			function RowClick(){
				console.log("Table Row Clicliclicked");
			    document.querySelector("#animalName").click();
			}
			
			var row_td = null;
			var row_dno = 0;
			
			function sendJet(data) {
				console.log("gonna publish~~~");
				row_td = data.getElementsByTagName("td");
				
				var orderData = {
					name : row_td[1].innerHTML,
					zone : row_td[2].innerHTML,
					num  : row_td[4].innerHTML
				};

				var jsonData = JSON.stringify(orderData); 
				console.log(jsonData);
				
				row_dno = orderData.num;
				console.log("출동 사건 번호: " + row_dno);
				
				message = new Paho.MQTT.Message(jsonData);
				message.destinationName = "/order/pub";
				client.send(message);
				console.log("order published");
				
				$("#finishSign").attr("src", "${pageContext.request.contextPath}/resource/img/complete.png");
				$("#beginSign").attr("src", "${pageContext.request.contextPath}/resource/img/begin2.png");
				$("#finishText").css('color', 'dimgray');
				$("#beginText").css('color', 'white');
				$("#beginText").attr("value", orderData.zone + " 구역 출동 중");		
				
				$("#numShow").attr("value", "사건 번호 " +  orderData.num + " 대응 중");
				$("#numShow").css('color', 'white');
				$("#zoneShow").attr("value", orderData.zone + " 구역에서");
				$("#zoneShow").css('color', 'white');
				$("#animalShow").attr("value", orderData.name + " 탐지 됨");
				$("#animalShow").css('color', 'white');
				
				//캔버스 맵에 "구역" 깃발 색 바꾸기
				if (orderData.zone == "A"){
					cctv1DetectingFlag = true;
					cctv2DetectingFlag = false;
					cctv3DetectingFlag = false;
					cctv4DetectingFlag = false;
				}
				else if (orderData.zone == "E"){
					cctv1DetectingFlag = false;
					cctv2DetectingFlag = true;
					cctv3DetectingFlag = false;
					cctv4DetectingFlag = false;
				}
				else if (orderData.zone == "K"){
					cctv1DetectingFlag = false;
					cctv2DetectingFlag = false;
					cctv3DetectingFlag = true;
					cctv4DetectingFlag = false;
				}
				else if (orderData.zone == "P"){
					cctv1DetectingFlag = false;
					cctv2DetectingFlag = false;
					cctv3DetectingFlag = false;
					cctv4DetectingFlag = true;
				}
			
				setTimeout(function() {
					$("#beginSign").attr("src", "${pageContext.request.contextPath}/resource/img/begin.png");
					$("#startSign").attr("src", "${pageContext.request.contextPath}/resource/img/arrived2.png");
					$("#beginText").css('color', 'dimgray');
					$("#startText").css('color', 'white');
					
					console.log("처리 중 실행");
				}, 2000);
				
				setTimeout(function() {
					$("#startSign").attr("src", "${pageContext.request.contextPath}/resource/img/arrived.png");
					$("#finishSign").attr("src", "${pageContext.request.contextPath}/resource/img/complete2.png");
					$("#startText").css('color', 'dimgray');
					$("#finishText").css('color', 'white');
					
					$("#numShow").attr("value", " --- ");
					$("#numShow").css('color', 'dimgray');
					$("#zoneShow").attr("value", " --- ");
					$("#zoneShow").css('color', 'dimgray');
					$("#animalShow").attr("value"," --- ");
					$("#animalShow").css('color', 'dimgray');
				
					$.ajax({
						type : 'post',
						dataType : 'json',
						data : {"dno" : row_dno},
						url: "${pageContext.request.contextPath}/home/dcompleteUpdate.do",
						async : false,
						success : 
							animalTable()
					});
					console.log("처리 완료");
					
					//캔버스 맵에서 "구역" 깃발 색 원래대로 돌리기
					cctv1DetectingFlag = false;
					cctv2DetectingFlag = false;
					cctv3DetectingFlag = false;
					cctv4DetectingFlag = false;
					
				}, 4000);
			
				setTimeout(function() {
					RowClick();
				}, 6000);
			
			}

			function response(index){
				//console.log(subList[index]+"에게 답장을 보내쥬");
				message = new Paho.MQTT.Message(ipid);
				message.destinationName = "/res/"+subList[index];
				client.send(message);
			}
			
			function onMessageArrived(message) {

				if(message.destinationName == "/order/ing"){
					/* 
					//출동 중 사인 내리고 처리 중 사인 올리기
					$("#beginSign").attr("src", "${pageContext.request.contextPath}/resource/img/begin.png");
					$("#startSign").attr("src", "${pageContext.request.contextPath}/resource/img/arrived2.png");
					$("#beginText").css('color', 'dimgray');
					$("#startText").css('color', 'white');
					 */
				}
				
				if(message.destinationName == "/order/completed"){
				/* 	
					console.log("order completed");
					//처리 중 사인 내리고 처리 완료 사인 올리기 + 텍스트 스타일 바꾸기
					$("#startSign").attr("src", "${pageContext.request.contextPath}/resource/img/arrived.png");
					$("#finishSign").attr("src", "${pageContext.request.contextPath}/resource/img/complete2.png");
					$("#startText").css('color', 'dimgray');
					$("#finishText").css('color', 'white');
					
					$("#numShow").attr("value", " --- ");
					$("#numShow").css('color', 'dimgray');
					$("#zonehow").attr("value", " --- ");
					$("#zonehow").css('color', 'dimgray');
					$("#animalShow").attr("value"," --- ");
					$("#animalShow").css('color', 'dimgray');

					$.ajax({
						type : 'post',
						dataType : 'json',
						data : {"dno" : row_dno},
						url: "${pageContext.request.contextPath}/home/dcompleteUpdate.do",
						success : 
							console.log("업데이트 성공!!!")
					});
					animalTable();
					RowClick();
					 */
				}

				if(message.destinationName =="/mirror") {
					const json = message.payloadString;
 					const obj = JSON.parse(json);
				
					if(obj.direction=="left"){
						location.href="history.do";
					} else if (obj.direction=="right"){
						location.href="analysis.do";
					}
				}

				if(message.destinationName =="/req/1jetracer") {
 					//const json = message.payloadString;
					//const obj = JSON.parse(json);
					//obj["witness"]= message.destinationName;
					
					$("#jrView1").attr("src", "data:image/jpg;base64,"+ message.payloadString);
/* 					
					if (obj.Class.length != 0){
						$("#j1Obj").attr("value", obj.Class);
						document.getElementById('j1Obj').style.color = '#DB6574';
						document.getElementById('j1Obj').style.fontWeight = 'bold';
						$("#j1Lev").attr("value", "등급이 몰까");
						document.getElementById('j1Lev').style.color = '#DB6574';
						document.getElementById('j1Lev').style.fontWeight = 'bold';
						$("#j1Loc").attr("value", "Jet 1 촬영 구간");
						document.getElementById('j1Loc').style.color = '#DB6574';
						document.getElementById('j1Loc').style.fontWeight = 'bold';
				
						if (obj["witness"].replace("/","") == "1jetracer"){
							document.getElementById('jrView1').style.border = '8px solid red';
						}
					}

					if (obj.Class.length == 0){
						
						$("#j1Obj").attr("value","*****  탐지대상 없음  *****");
						document.getElementById('j1Obj').style.color = 'white';
						$("#j1Lev").attr("value","*****  해당사항 없음  *****");
						document.getElementById('j1Lev').style.color = 'white';
						$("#j1Loc").attr("value","*****  해당사항 없음  *****");
						document.getElementById('j1Loc').style.color = 'white';
						document.getElementById('jrView1').style.border = 'inactiveborder';
					}
					 */
				}
				
				if(message.destinationName =="/req/2jetracer") {
 					//const json = message.payloadString;
					//const obj = JSON.parse(json);
					//obj["witness"]= message.destinationName;
					
					$("#jrView2").attr("src", "data:image/jpg;base64,"+ message.payloadString);
/* 					
					if (obj.Class.length != 0){
						$("#j2Obj").attr("value", obj.Class);
						document.getElementById('j2Obj').style.color = '#DB6574';
						document.getElementById('j2Obj').style.fontWeight = 'bold';
						$("#j2Lev").attr("value", "등급이 몰까");
						document.getElementById('j2Lev').style.color = '#DB6574';
						document.getElementById('j2Lev').style.fontWeight = 'bold';
						$("#j2Loc").attr("value", "Jet 1 촬영 구간");
						document.getElementById('j2Loc').style.color = '#DB6574';
						document.getElementById('j2Loc').style.fontWeight = 'bold';
				
						if (obj["witness"].replace("/","") == "1jetracer"){
							document.getElementById('jrView2').style.border = '8px solid red';
						}
					}

					if (obj.Class.length == 0){
						
						$("#j2Obj").attr("value","*****  탐지대상 없음  *****");
						document.getElementById('j2Obj').style.color = 'white';
						$("#j2Lev").attr("value","*****  해당사항 없음  *****");
						document.getElementById('j2Lev').style.color = 'white';
						$("#j2Loc").attr("value","*****  해당사항 없음  *****");
						document.getElementById('j2Loc').style.color = 'white';
						document.getElementById('jrView2').style.border = 'inactiveborder';
					}
					 */
				}
				
				if(message.destinationName =="/req/3jetracer") {
 					//const json = message.payloadString;
					//const obj = JSON.parse(json);
					//obj["witness"]= message.destinationName;
					
					$("#jrView3").attr("src", "data:image/jpg;base64,"+ message.payloadString);
/* 					
					if (obj.Class.length != 0){
						$("#j2Obj").attr("value", obj.Class);
						document.getElementById('j2Obj').style.color = '#DB6574';
						document.getElementById('j2Obj').style.fontWeight = 'bold';
						$("#j2Lev").attr("value", "등급이 몰까");
						document.getElementById('j2Lev').style.color = '#DB6574';
						document.getElementById('j2Lev').style.fontWeight = 'bold';
						$("#j2Loc").attr("value", "Jet 1 촬영 구간");
						document.getElementById('j2Loc').style.color = '#DB6574';
						document.getElementById('j2Loc').style.fontWeight = 'bold';
				
						if (obj["witness"].replace("/","") == "1jetracer"){
							document.getElementById('jrView2').style.border = '8px solid red';
						}
					}

					if (obj.Class.length == 0){
						
						$("#j2Obj").attr("value","*****  탐지대상 없음  *****");
						document.getElementById('j2Obj').style.color = 'white';
						$("#j2Lev").attr("value","*****  해당사항 없음  *****");
						document.getElementById('j2Lev').style.color = 'white';
						$("#j2Loc").attr("value","*****  해당사항 없음  *****");
						document.getElementById('j2Loc').style.color = 'white';
						document.getElementById('jrView2').style.border = 'inactiveborder';
					}
					 */
				}
				
 				if(message.destinationName =="/req/1cctv") {
 					response(3);
					lastSendtimearr[3] = Date.now();
 					
 					const json = message.payloadString;
					const obj = JSON.parse(json);
					obj["witness"]= message.destinationName;
					
					$("#cameraView1").attr("src", "data:image/jpg;base64,"+ obj.Cam);
					
					if (obj.Class.length != 0){
						
						$("#c1Obj").attr("value", obj.Class);
						document.getElementById('c1Obj').style.color = '#ADFF2F';
						document.getElementById('c1Col1').style.color = '#ADFF2F';
						document.getElementById('c1Obj').style.fontWeight = 'bold';
						$("#c1Lev").attr("value", "등급이 몰까");
						document.getElementById('c1Lev').style.color = '#ADFF2F';
						document.getElementById('c1Lev').style.fontWeight = 'bold';
						$("#c1Loc").attr("value", "1번 CCTV 촬영 구간");
						document.getElementById('c1Loc').style.color = '#ADFF2F';
						document.getElementById('c1Loc').style.fontWeight = 'bold';
				
						if (obj["witness"].replace("/","") == "req1cctv"){
							document.getElementById('cameraView1').style.border = '8px solid red';
						}
					}

					if (obj.Class.length == 0){
						
						$("#c1Obj").attr("value","*****  탐지대상 없음  *****");
						document.getElementById('c1Col1').style.color = 'white';
						document.getElementById('c1Obj').style.color = 'white';
						$("#c1Lev").attr("value","*****  해당사항 없음  *****");
						document.getElementById('c1Lev').style.color = 'white';
						$("#c1Loc").attr("value","*****  해당사항 없음  *****");
						document.getElementById('c1Loc').style.color = 'white';
						document.getElementById('cameraView1').style.border = 'inactiveborder';
					}
				}
 				
				if(message.destinationName =="/req/2cctv") {
					response(4);
					lastSendtimearr[4] = Date.now();
					
					const json = message.payloadString;
					const obj = JSON.parse(json);
					obj["witness"]= message.destinationName;

					$("#cameraView2").attr("src", "data:image/jpg;base64,"+ obj.Cam);
					
					if (obj.Class.length != 0){
						
						$("#c2Obj").attr("value", obj.Class);
						document.getElementById('c2Col1').style.color = '#ADFF2F';
						document.getElementById('c2Obj').style.color = '#ADFF2F';
						document.getElementById('c2Obj').style.fontWeight = 'bold';
						$("#c2Lev").attr("value", "등급이 몰까");
						document.getElementById('c2Lev').style.color = '#ADFF2F';
						document.getElementById('c2Lev').style.fontWeight = 'bold';
						$("#c2Loc").attr("value", "2번 CCTV 촬영 구간");
						document.getElementById('c2Loc').style.color = '#ADFF2F';
						document.getElementById('c2Loc').style.fontWeight = 'bold';
						document.getElementById('cameraView2').style.border = '8px solid red';
					}
					
					if (obj.Class.length == 0){
						document.getElementById('c2Col1').style.color = 'white';
						$("#c2Obj").attr("value","*****  탐지대상 없음  *****");
						document.getElementById('c2Obj').style.color = 'white';
						$("#c2Lev").attr("value","*****  해당사항 없음  *****");
						document.getElementById('c2Lev').style.color = 'white';
						$("#c2Loc").attr("value","*****  해당사항 없음  *****");
						document.getElementById('c2Loc').style.color = 'white';
						document.getElementById('cameraView2').style.border = 'inactiveborder';
					}
				}

				if(message.destinationName =="/req/3cctv") {
					response(5);
					lastSendtimearr[5] = Date.now();
					
					const json = message.payloadString;
					const obj = JSON.parse(json);
					obj["witness"]= message.destinationName;
					
					$("#cameraView3").attr("src", "data:image/jpg;base64,"+ obj.Cam);
					
					if (obj.Class.length != 0){
						$("#c3Obj").attr("value", obj.Class);
						document.getElementById('c3Col1').style.color = '#ADFF2F';
						document.getElementById('c3Obj').style.color = '#ADFF2F';
						document.getElementById('c3Obj').style.fontWeight = 'bold';
						$("#c3Lev").attr("value", "등급이 몰까");
						document.getElementById('c3Lev').style.color = '#ADFF2F';
						document.getElementById('c3Lev').style.fontWeight = 'bold';
						$("#c3Loc").attr("value", "3번 CCTV 촬영 구간");
						document.getElementById('c3Loc').style.color = '#ADFF2F';
						document.getElementById('c3Loc').style.fontWeight = 'bold';
						document.getElementById('cameraView3').style.border = '8px solid red';
					}

					if (obj.Class.length == 0){
						document.getElementById('c3Col1').style.color = 'white';
						$("#c3Obj").attr("value","*****  탐지대상 없음  *****");
						document.getElementById('c3Obj').style.color = 'white';
						$("#c3Lev").attr("value","*****  해당사항 없음  *****");
						document.getElementById('c3Lev').style.color = 'white';	
						$("#c3Loc").attr("value","*****  해당사항 없음  *****");
						document.getElementById('c3Loc').style.color = 'white';
						document.getElementById('cameraView3').style.border = 'inactiveborder';
					}
				}

				if(message.destinationName =="/req/4cctv") {
					response(6);
					lastSendtimearr[6] = Date.now();
					
					const json = message.payloadString;
					const obj = JSON.parse(json);
					obj["witness"]= message.destinationName;

					$("#cameraView4").attr("src", "data:image/jpg;base64,"+ obj.Cam);
					
					if (obj.Class.length != 0){
						$("#c4Obj").attr("value", obj.Class);
						document.getElementById('c4Col1').style.color = '#ADFF2F';
						document.getElementById('c4Obj').style.color = '#ADFF2F';
						document.getElementById('c4Obj').style.fontWeight = 'bold';
						$("#c4Lev").attr("value", "등급이 몰까");
						document.getElementById('c4Lev').style.color = '#ADFF2F';
						document.getElementById('c4Lev').style.fontWeight = 'bold';
						$("#c4Loc").attr("value", "4번 CCTV 촬영 구간");
						document.getElementById('c4Loc').style.color = '#ADFF2F';
						document.getElementById('c4Loc').style.fontWeight = 'bold';
						document.getElementById('cameraView4').style.border = '8px solid red';
					}
					
					if (obj.Class.length == 0){
						$("#c4Obj").attr("value","*****  탐지대상 없음  *****");
						document.getElementById('c4Col1').style.color = 'white';
						document.getElementById('c4Obj').style.color = 'white';
						document.getElementById('c4Obj').style.fontWeight = 'normal';
						document.getElementById('c4Obj').style.opacity = '0.9';
						$("#c4Lev").attr("value","*****  해당사항 없음  *****");
						document.getElementById('c4Lev').style.color = 'white';
						document.getElementById('c4Lev').style.fontWeight = 'normal';
						$("#c4Loc").attr("value","*****  해당사항 없음  *****");
						document.getElementById('c4Loc').style.color = 'white';
						document.getElementById('c4Loc').style.fontWeight = 'normal';
						document.getElementById('cameraView4').style.border = 'inactiveborder'; 
					}
				}
			}
			
			var flagarr=[];
	        function startGame() {
	            car1 = new component(15, 20, "red", 50, 350, Math.PI); // component 생성
	            car2 = new component(15, 20, "green", 300, 50, Math.PI / 2);
	            car3 = new component(15, 20, "blue", 350, 50, Math.PI / 2);
	            
	            flagA = new component(15, 20, "A", 400, 50, Math.PI / 2);
	            flagB = new component(15, 20, "B", 315, 50, Math.PI / 2);
	            flagC = new component(15, 20, "C", 230, 50, Math.PI / 2);
	            flagD = new component(15, 20, "D", 150, 50, Math.PI / 2);
	            flagE = new component(15, 20, "E", 100, 100, Math.PI / 2);
	            flagF = new component(15, 20, "F", 100, 150, Math.PI / 2);
	            flagH = new component(15, 20, "H", 50, 300, Math.PI / 2);
	            flagI = new component(15, 20, "I", 50, 350, Math.PI / 2);
	            flagJ = new component(15, 20, "J", 100, 400, Math.PI / 2);
	            flagK = new component(15, 20, "K", 150, 450, Math.PI / 2);
	            flagM = new component(15, 20, "M", 275, 450, Math.PI / 2);
	            flagN = new component(15, 20, "N", 400, 450, Math.PI / 2);
	            flagP = new component(15, 20, "P", 450, 310, Math.PI / 2);
	            flagS = new component(15, 20, "S", 450, 205, Math.PI / 2);
	            flagT = new component(15, 20, "T", 450, 100, Math.PI / 2);
	            
	            flagarr=[flagA,flagB,flagC,flagD,flagE,flagF,flagH,flagI,flagJ,flagK,flagM,flagN,flagP,flagS,flagT]
	            myGameArea.start(); 
	        }
	        function stop() { // 테스트 용
	           clearInterval(myGameArea.interval);
	        }
	        //var scale = myGameArea.canvas.width / 500;
	        var myGameArea = {
	            canvas : document.createElement("canvas"), // 캔버스 태그 생성
	            start : function() {
	                this.canvas.width = "440"; // 캔버스 크기 설정
	                this.canvas.height = "440";
	        		this.canvas.style.position = "absolute";
	                this.canvas.style.left= "40px";
	                this.canvas.style.top= "40px";
	                this.canvas.style.bottom= "0";
	                this.scale = this.canvas.width / 500;
	                this.context = this.canvas.getContext("2d"); // 캔버스에서 그리기 도구 객체 얻기
	                this.context.scale(this.scale, this.scale);
	                var canvashere=document.getElementById("canvashere")
	                canvashere.insertBefore(this.canvas, canvashere.childNodes[0]); // 캔버스 태그를 body 태그 안에 넣기
	                this.interval = setInterval(updateGameArea, 20); // 20ms마다 updateGameArea 함수를 실행
	            },
	            stop : function() {
	                clearInterval(this.interval);
	            },    
	            clear : function() {
	                this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
	            }
	        }
	        function component(width, height, color, x, y, angle) {
	            this.width = width;
	            this.height = height;
	            this.speed = 0;
	            this.angle = angle; // 현대 각도
	            this.moveAngle = 0; // 변화 각도
	            this.x = x; // component 좌표
	            this.y = y;
	            this.color= color;
	            this.update = function() {
	                   ctx = myGameArea.context; // 그리기 객체 도구 얻기
	                   
	                   // component rotation에 필요
	                   ctx.save();
	                   ctx.translate(this.x, this.y);
	                   ctx.rotate(this.angle);
	                   
	                   
	                   //ctx.fillStyle = color; // 직사각형 색 설정
	                   var image= document.createElement("img");
	                   image.src="${pageContext.request.contextPath}/resource/img/myjetracer.png";
	                   ctx.drawImage(image, -20, -20);
	                   //ctx.fillRect(this.width / -2, this.height / -2, this.width, this.height); // 직사각형 그리기
	                   
	                   // component rotation을 위해 바꾼 설정을 복구
	                   ctx.restore();
	            }
	            this.update2 = function() {
	                ctx = myGameArea.context;
	                ctx.save();
	                ctx.translate(this.x, this.y);
	                ctx.fillStyle = "rgba(255, 255, 0, 0.8)";
	                ctx.fillRect(this.width / -2, this.height / -2, 5, 10);
	     			ctx.fillStyle = "rgba(0, 128, 0, 1)";
	     			ctx.fillRect(this.width / -2, this.height / -2-30, 40, 30);
	     			ctx.font = "30px Arial";
	     			ctx.fillStyle = "white";
	     			ctx.fillText(this.color, this.width /-2+10, this.height / -2-5);
	                ctx.restore();
	         	}
	            this.update3 = function() {
	                ctx = myGameArea.context;
	                ctx.save();
	                ctx.translate(this.x, this.y);
	                ctx.fillStyle = "rgba(255, 255, 0, 0.8)";
	                ctx.fillRect(this.width / -2, this.height / -2, 5, 10);
	     			ctx.fillStyle = "rgba(255, 0, 0, 1)";
	     			ctx.fillRect(this.width / -2, this.height / -2-30, 40, 30);
	     			ctx.font = "30px Arial";
	     			ctx.fillStyle = "white";
	     			ctx.fillText(this.color, this.width /-2+10, this.height / -2-5);
	                ctx.restore();
	         	}
	           
	            this.fixPosition2 = function() {
	               if(this.x > 150 && this.x <= 400 && this.y == 50) {
	                   this.moveLeft();}
	               if(this.x <= 150 && this.y >= 50 && this.y <100) {
	                     this.turnLeft2();}
	               if(this.x == 100 && this.y >= 100 && this.y <150) {
	                     this.moveDown();}
	               if(this.y >= 150 && this.y <300 && this.angle >= Math.PI && this.angle <= Math.PI + 0.62 * Math.PI / 6) {
	                     this.moveDown2();}
	               if(this.x == 50 && this.y >= 300 && this.y < 350) {
	                     this.moveDown();}
	               if(this.x >= 50 && this.x < 100 && this.y >= 350 && this.angle >= Math.PI / 2 && this.angle <= Math.PI) {
	                     this.turnLeft3();}
	               if(this.x >= 100 && this.x < 125 && this.y >= 400 && this.y < 425 && this.angle >= Math.PI / 2 && this.angle <= Math.PI) {
	                     this.turnRight6();}
	               if(this.x >= 125 && this.x < 150 && this.y >= 425 && this.y < 450 && this.angle >= Math.PI / 2 && this.angle <= Math.PI) {
	                   this.turnLeft4();}
	               if(this.x >= 150 && this.x < 400 && this.y == 450) {
	                   this.moveRight();}
	               if(this.x >= 400 && this.angle > 0 && this.angle <= 3 * Math.PI / 2) {
	                   this.turnLeft5();}
	               
	               if(this.x == 450 && this.y > 100 && this.y <= 400) {
	                   this.moveUp1();}
	               if(this.x > 400 && this.x <= 450 && this.y <= 100 && this.angle >= 2 * Math.PI / 2 && this.angle <= 5 * Math.PI / 2) { 
	                   this.turnLeft6();}
	            }
	            
	            this.moveRight = function() {
	               this.speed = 1;
	                this.angle = Math.PI / 2;
	                this.x += this.speed * Math.sin(this.angle);
	                this.y -= this.speed * Math.cos(this.angle);
	            }
	            
	            this.turnRight1 = function() {
	               this.speed = 1;
	               this.moveAngle = 1.145;
	                this.angle +=  this.moveAngle * Math.PI / 180;
	                if (this.angle >= Math.PI) { this.x = 450; this.y = 100; return;}
	                this.x += this.speed * Math.sin(this.angle);
	                this.y -= this.speed * Math.cos(this.angle);
	            }
	            
	            this.moveDown = function() {
	               this.speed = 1;
	                this.angle = Math.PI;
	                this.x += this.speed * Math.sin(this.angle);
	                this.y -= this.speed * Math.cos(this.angle);
	            }
	            
	            this.turnRight2 = function() {
	               this.speed = 1;
	               this.moveAngle =  1.145;
	                this.angle +=  this.moveAngle * Math.PI / 180;
	                if (this.angle >= 3 * Math.PI / 2) { this.x = 400; this.y = 450; return;}
	                this.x += this.speed * Math.sin(this.angle);
	                this.y -= this.speed * Math.cos(this.angle);
	            }
	            
	            this.moveLeft = function() {
	               this.speed = 1;
	                this.angle = 3 * Math.PI / 2;
	                this.x += this.speed * Math.sin(this.angle);
	                this.y -= this.speed * Math.cos(this.angle);
	            }
	            
	            this.turnRight3 = function() {
	               this.speed = 1;
	               this.moveAngle = 2;
	                this.angle +=  this.moveAngle * Math.PI / 180;
	                if (this.angle >= 11 * Math.PI / 6) {this.x = 135; this.y = 435; return;} //this.x = 135; this.y = 435; return;
	                this.x += this.speed * Math.sin(this.angle);
	                this.y -= this.speed * Math.cos(this.angle);
	            }
	            
	            this.turnLeft = function() {
	               this.speed = 1;
	               this.moveAngle = -0.7;
	                this.angle +=  this.moveAngle * Math.PI / 180;
	                if (this.x < 101) {this.x = 100; this.y = 400; this.angle = 3 * Math.PI / 2; return;} // this.x = 100; this.y = 400; return
	                this.x += this.speed * Math.sin(this.angle);
	                this.y -= this.speed * Math.cos(this.angle);
	            }
	            
	            this.turnRight4 = function() {
	               this.speed = 1;
	               this.moveAngle = 1.145;
	                this.angle +=  this.moveAngle * Math.PI / 180;
	                if (this.angle >= 2 * Math.PI) {this.x = 50; this.y = 350; return;}
	                this.x += this.speed * Math.sin(this.angle);
	                this.y -= this.speed * Math.cos(this.angle);
	            }
	            
	            this.moveUp1 = function() {
	               this.speed = 1;
	                this.angle =  2 * Math.PI;
	                this.x += this.speed * Math.sin(this.angle);
	                this.y -= this.speed * Math.cos(this.angle);
	            }
	            
	            this.moveUp2 = function() {
	               this.speed = 1;
	                this.angle = 12.62 * Math.PI / 6;
	                this.x += this.speed * Math.sin(this.angle);
	                this.y -= this.speed * Math.cos(this.angle);
	                if(this.y <= 150) {this.x = 100; this.y = 150; return;} //this.x = 100; this.y = 150; return
	            }
	            
	            this.turnRight5 = function() {
	               this.speed = 1;
	               this.moveAngle = 1.145;
	                this.angle +=  this.moveAngle * Math.PI / 180;
	                if (this.angle >= 5 * Math.PI / 2) { this.x = 150; this.y = 50; return}
	                this.x += this.speed * Math.sin(this.angle);
	                this.y -= this.speed * Math.cos(this.angle);
	            }
	            
	            this.turnLeft2 = function() {
	                this.speed = 1;
	                this.moveAngle = -1.145;
	                 this.angle +=  this.moveAngle * Math.PI / 180;
	                 if (this.angle <= Math.PI) { this.x = 100; this.y = 100; return}
	                 this.x += this.speed * Math.sin(this.angle);
	                 this.y -= this.speed * Math.cos(this.angle);
	             }
	            
	            this.moveDown2 = function() {
	                this.speed = 1;
	                 this.angle = Math.PI + 0.62 * Math.PI / 6;
	                 this.x += this.speed * Math.sin(this.angle);
	                 this.y -= this.speed * Math.cos(this.angle);
	                 if(this.y >= 300) {this.x = 50; this.y = 300; return;} //this.x = 100; this.y = 150; return
	             }
	            
	            this.turnLeft3 = function() {
	                this.speed = 1;
	                this.moveAngle = -1.145;
	                 this.angle +=  this.moveAngle * Math.PI / 180;
	                 if (this.angle <= Math.PI / 2) {this.x = 100; this.y = 400; this.angle = Math.PI / 2; return;}
	                 this.x += this.speed * Math.sin(this.angle);
	                 this.y -= this.speed * Math.cos(this.angle);
	             }
	            
	            this.turnRight6 = function() {
	                this.speed = 1;
	                this.moveAngle = 2.25;
	                 this.angle +=  this.moveAngle * Math.PI / 180;
	                 if (this.angle >= Math.PI) {this.x = 125; this.y = 425; this.angle = Math.PI; return;} // this.x = 100; this.y = 400; return
	                 this.x += this.speed * Math.sin(this.angle);
	                 this.y -= this.speed * Math.cos(this.angle);
	             }
	            
	            this.turnLeft4 = function() {
	               this.speed = 1;
	               this.moveAngle = -2.25;
	                this.angle +=  this.moveAngle * Math.PI / 180;
	                if (this.angle <= Math.PI / 2) {this.x = 150; this.y = 450; this.angle = Math.PI / 2; return;} //this.x = 135; this.y = 435; return;
	                this.x += this.speed * Math.sin(this.angle);
	                this.y -= this.speed * Math.cos(this.angle);
	            }
	            
	            this.turnLeft5 = function() {
	                this.speed = 1;
	                this.moveAngle = -1.145;
	                 this.angle +=  this.moveAngle * Math.PI / 180;
	                 if (this.angle <= 0) {this.x = 450; this.y = 400; this.angle = 0; return;}
	                 this.x += this.speed * Math.sin(this.angle);
	                 this.y -= this.speed * Math.cos(this.angle);
	             }
	            
	            this.turnLeft6 = function() {
	                this.speed = 1;
	                this.moveAngle = -1.145;
	                 this.angle +=  this.moveAngle * Math.PI / 180;
	                 if (this.angle <= 3 * Math.PI / 2) {this.x = 400; this.y = 50; this.angle = 3 * Math.PI / 2; return;}
	                 this.x += this.speed * Math.sin(this.angle);
	                 this.y -= this.speed * Math.cos(this.angle);
	             }
	            this.crashWith = function(otherobj) {
	        		var myx = this.x;
	        		var myy = this.y;
	        		var otherx = otherobj.x;
	        		var othery = otherobj.y;
	                var interx= Math.abs(myx-otherx);
	                var intery= Math.abs(myy-othery);
	                var interrad = 5;
	        		var crash = true;
	        		if (((interx*interx)+(intery*intery))>(interrad*interrad)) {
	            		crash = false;
	        		}
	        		return crash;
				}
				            
	            this.crashWitharr = function(otherobjarr) {
	        		var myx = this.x;
	        		var myy = this.y;
	        		var crash = true;
	        		function checkoth(otherobj) {
	            		var otherx = otherobj.x;
	            		var othery = otherobj.y;
	                    var interx= Math.abs(myx-otherx);
	                    var intery= Math.abs(myy-othery);
	                    var interrad = 5;
	            		return ((interx*interx)+(intery*intery))<(interrad*interrad);
	                }
	        		crash = otherobjarr.some(checkoth)
	        		return crash;
				}
	            
	        }
	       	
	        function updateGameArea() {
	           myGameArea.clear(); // 캔버스 지우기
	            drawMap(); //지도 그리기
	            
	            car1.moveAngle = 0; // 변화 각도 초기화
	            car1.speed = 0; // 속도 초기화
	            car2.moveAngle = 0;
	            car2.speed = 0;
	            car3.moveAngle = 0;
	            car3.speed = 0;
	            
	            // component 위치 조정
	            // 움직임 통제, 방향 통제 flag를 만족하면 위치 조정
	            console.log(car1.crashWitharr(flagarr));
				/* if (car1.crashWitharr(flagarr)) {
				car1.fixPosition2();
	            } else {
				car1.fixPosition2();
	            } */
	            car1.fixPosition2();
	            car2.fixPosition2();
	            car3.fixPosition2();
	                
	            car1.update(); // component 그리기
	            car2.update();
	            car3.update();
	            
	            if (cctv1DetectingFlag){
	            	flagA.update3();
	            } else if (!cctv1DetectingFlag) {
	            	flagA.update2();
	            }
	            
	            flagB.update2();
	            flagC.update2();
	            flagD.update2();
	            
	            if (cctv2DetectingFlag){
	            	flagE.update3();
	            } else if (!cctv2DetectingFlag){
	            	flagE.update2();
	            }
	            
	            flagF.update2();
	            flagH.update2();
	            flagI.update2();
	            flagJ.update2();
	            
	            if(cctv3DetectingFlag){
	            	flagK.update3();
	            } else if (!cctv3DetectingFlag){
	            	flagK.update2();
	            }

	            flagM.update2();
	            flagN.update2();
	            
	            if(cctv4DetectingFlag){
	            	flagP.update3();
	            } else if (!cctv4DetectingFlag){
	            	flagP.update2();
	            }
	            
	            flagS.update2();
	            flagT.update2();
	            
	            //console.log(car1.x, car1.y, car1.angle * 180 / Math.PI);
	        }
	        function drawMap() {
	             var ctx = myGameArea.context; // 캔버스에서 그리기 도구 객체 얻기
	             
	             // 도로 테두리 그리기
	             ctx.beginPath(); // path 시작 함수, path를 초기화 또는 재설정
	             ctx.lineWidth = ' 52' // path의 굴기 설정
	             ctx.strokeStyle = "white"; // path의 색 설정
	             ctx.moveTo(150, 50); // path의 시작점
	             ctx.lineTo(400, 50); // 해당 좌표로 직선 이어주기
	             ctx.arcTo(450, 50, 450, 100, 50); // 해당 좌표로 곡선 이어주기
	             ctx.lineTo(450, 400);
	             ctx.arcTo(450, 450, 400, 450, 50);
	             ctx.lineTo(150, 450);
	             ctx.bezierCurveTo(130, 450, 130, 400, 100, 400); // 해당 좌표로 bezier curve 이어주기
	             ctx.arcTo(50, 400, 50, 350, 50);
	             ctx.lineTo(50, 300);
	             ctx.lineTo(100, 150);
	             ctx.lineTo(100, 100);
	             ctx.arcTo(100, 50, 150, 50, 50);
	             ctx.stroke(); // 위에서 이어준 좌표 실제로 그리기
	             
	             // 도로 내부 그리기
	             ctx.beginPath();
	             ctx.lineWidth = "50";
	             ctx.strokeStyle = "black";
	             ctx.moveTo(150, 50);
	             ctx.lineTo(400, 50);
	             ctx.arcTo(450, 50, 450, 100, 50);
	             ctx.lineTo(450, 400);
	             ctx.arcTo(450, 450, 400, 450, 50);
	             ctx.lineTo(150, 450);
	             ctx.bezierCurveTo(130, 450, 130, 400, 100, 400);
	             ctx.arcTo(50, 400, 50, 350, 50);
	             ctx.lineTo(50, 300);
	             ctx.lineTo(100, 150);
	             ctx.lineTo(100, 100);
	             ctx.arcTo(100, 50, 150, 50, 50);
	             ctx.stroke();
	             // 도로 중앙선 그리기
	             ctx.beginPath();
	             ctx.lineWidth = "1";
	             ctx.strokeStyle = "white";
	             ctx.setLineDash([15, 20]) // path를 점선으로 설정, 길이 15, 간격 20
	             ctx.moveTo(150, 50);
	             ctx.lineTo(400, 50);
	             ctx.arcTo(450, 50, 450, 100, 50);
	             ctx.lineTo(450, 400);
	             ctx.arcTo(450, 450, 400, 450, 50);
	             ctx.lineTo(150, 450);
	             ctx.bezierCurveTo(130, 450, 130, 400, 100, 400);
	             ctx.arcTo(50, 400, 50, 350, 50);
	             ctx.lineTo(50, 300);
	             ctx.lineTo(100, 150);
	             ctx.lineTo(100, 100);
	             ctx.arcTo(100, 50, 150, 50, 50);
	             ctx.stroke();
	             ctx.setLineDash([]); // path를 실선으로 초기화, 이렇게 해야 다음에 또 그릴 때 위의 코드가 점선으로 시작 안함
	        }
		</script>
	</head>
	
	<body onload="startGame()">
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
	          <div class="avatar" style="width: 100px; height: 100px; align-itself: center; "><img src="${pageContext.request.contextPath}/resource/img/milk.jpg" class="img-fluid rounded-circle"></div>
	          <div class="title">
	            <h1 class="h5" style="color: lightgray">AIoT Project</h1>
	            <p style="color: lightgray">관리자</p>
	          </div>
	        </div>
	        <span class="heading" style="color:#DB6574">CATEGORIES</span>
	         <ul class="list-unstyled">
	          <li><a href="${pageContext.request.contextPath}/home/main.do" style="color: lightgray"> <i class="icon-home"></i>메인 페이지</a></li>
	          <li><a href="${pageContext.request.contextPath}/home/jetracer.do" style="color: lightgray"> <i class="icon-writing-whiteboard"></i>탐지봇 현황 </a></li>
	          <li><a href="${pageContext.request.contextPath}/home/history.do" style="color: lightgray"> <i class="icon-grid"></i>탐지 히스토리 조회 </a></li>
	          <li class="active"><a href="${pageContext.request.contextPath}/home/status.do" style="color: lightgray"> <i class="icon-padnote"></i>실시간 탐지 | 대응 현황</a></li>
	      	  <li><a href="${pageContext.request.contextPath}/home/analysis.do" style="color: lightgray"> <i class="icon-chart"></i>탐지 결과 분석 </a></li>
	      	 </ul>
	      </nav>

	     <div class="page-content" style="top: -50px;height: 1080px; padding-bottom: 0px; ">
	     	
	     	<div style="margin-bottom: 10px; margin-top: 60px; color: white; font-weight: bold; font-size: xx-large; height: 50px; width: 1623px; text-align: center;">실시간 유해동물 탐지 및 대응 현황</div>
		     
		     <section style="padding-right: 0px">
	          <div class="container-fluid">
	         	<div class="container" style="position:absolute; margin-right: 0px; margin-left: 0px; width: 520px; height: 468px; top: 130px">
	         	  <input value="JetRacer 탐지 현황" readonly="readonly" style="background-color: #ADFF2F;; color: black; font-weight: 500; font-size:20px; border-color: transparent; font-weight: bold; width: 520px; text-align: center; margin-left: -15px"/>
				  <div class="row row-cols-2">
				    <div class="col" style="padding-left: 0px; padding-right: 0px; width: 260px; height: 220px"><img id=jrView1 style="width: 260px; height: 220px; padding-left: 0px; padding-right: 0px"/></div>
				    <div class="col" style="padding-left: 0px; padding-right: 0px; width: 260px; height: 220px"><img id=jrView2 style="width: 260px; height: 220px; padding-left: 0px; padding-right: 0px"/></div>
				    <div class="col" style="padding-left: 0px; padding-right: 0px; width: 260px; height: 220px"><img id=jrView3 style="width: 260px; height: 220px; padding-left: 0px; padding-right: 0px"/></div>
				    <div class="col" style="padding-left: 0px; padding-right: 0px; width: 260px; height: 220px"><img src="${pageContext.request.contextPath}/resource/img/jetracer.jpg" style="width: 260px; height: 220px; opacity: 0.5; margin-left: 0px; margin-right: 0px"/></div>
				  </div>
				</div>
	          </div>
	        </section>
	        
	        <section style="padding-right: 0px">
	          <div class="container-fluid">
	         	<div class="container" style="position:absolute; margin-right: 0px; margin-left: 520px; width: 520px; height: 468px; top: 130px">
	         	  <input value="CCTV 탐지 현황" readonly="readonly" style="background-color: #ADFF2F; color: black; font-weight: 500; font-size:20px; border-color: transparent; font-weight: bold;width: 520px; text-align: center; margin-left: -15px"/>
				  <div class="row row-cols-2">
				    <div class="col" style="padding-left: 0px; padding-right: 0px; width: 260px; height: 220px"><img id=cameraView1 style="width: 260px; height: 220px; padding-left: 0px; padding-right: 0px; border:inactiveborder; "/></div>
				    <div class="col" style="padding-left: 0px; padding-right: 0px; width: 260px; height: 220px"><img id=cameraView2 style="width: 260px; height: 220px; padding-left: 0px; padding-right: 0px; borderstyle: none; bordercolor: transparent; borderwidth: inherit"/></div>
				    <div class="col" style="padding-left: 0px; padding-right: 0px; width: 260px; height: 220px"><img id=cameraView3 style="width: 260px; height: 220px; padding-left: 0px; padding-right: 0px; borderstyle: none; bordercolor: transparent; borderwidth: inherit"/></div>
				    <div class="col" style="padding-left: 0px; padding-right: 0px; width: 260px; height: 220px"><img id=cameraView4 style="width: 260px; height: 220px; padding-left: 0px; padding-right: 0px; borderstyle: none; bordercolor: transparent; borderwidth: inherit"/></div>
				  </div>
				</div>
	          </div>
	        </section>
	       
	        <section style="padding-right: 0px">
	          <div class="container-fluid">
	         	<div class="container" style="position:absolute; margin-right: 0px; margin-left: 1040px; width: 520px; height: 468px; top: 130px; ">
	         	  <input value="유해동물 탐지 위치" readonly="readonly" style="background-color: #864DD9; color: white; font-weight: 500; font-size:20px;border-color: transparent; font-weight: bold; width: 520px; text-align: center;"/>
				   <div id="canvashere"style="background-color:transparent ; width: 520px; height: 440px; color: white;text-align: center ;font-size: xx-large; justify-content: center; border-color: #864DD9; border-style:solid; border-width:medium;"></div>
				</div>
	          </div>
	        </section>
	        
	        <input value="CCTV가 탐지한 유해동물 리스트"readonly="readonly" style="background-color: #864DD9; color: white; font-weight: 500; font-size:20px; margin-left: 1085px ;border-color: transparent; font-weight: bold;position: absolute;margin-top:500px; height: 36px; padding: 0px; text-align: center;width: 520px"/>
       		
       		<div style="background-color: transparent ; height: 405px; width: 520px; margin-top:535px; margin-left:1085px; padding:0px ; position: absolute">
      			
     			<div class="table-responsive" style=" border-color: #864DD9; border-style:solid; border-width:medium; padding-left:5px; padding-right:5px;">
     			  <div  id="animalTable" style=" text-align: center; justify-content: center; height: 150px">
                   <table class="table table-striped table-sm" style="color: white; height: 150px">
                     <thead style="border-style:double ; border-left: hidden; border-right: hidden; border-top: hidden; border-color: #864DD9; font-size: medium;">
                       <tr style="height: 40px; justify-content: center; padding:0px; color:#864DD9; text-align: center; ">
                         <th style="width: 100px">CCTV #</th>
                         <th style="width: 160px">유해동물 이름</th> 
                         <th style="width: 80px">탐지 구역</th>
                         <th style="width: 140px">탐지 시각</th>
                         <th style="width: 40px; display: none" >Dno</th>
                       </tr>
                     </thead>
                     <tbody style="font-size: medium" id="detectedAnimal">
                      <c:forEach var="animal" items="${animal}">
                      	<tr onclick="sendJet(this)">
                      	  <td scope="row">${animal.dfinder}</td>
                          <td id="animalName">${animal.dname}</td>
                          <td>${animal.dzone}</td>
                          <td><fmt:formatDate value="${animal.dtime}" pattern="MM/dd HH:mm:ss"/></td>
                          <td style="display: none;">${animal.dno}</td>
                        </tr>
                     </c:forEach>
                    </tbody>
                   </table>
                 </div>
               </div>

               <div style="height: 239px; background-color : transparent ; text-align: center; justify-content: center; margin-top: 10px; border-color: #864DD9; border-style:solid; border-width:medium; ">
               		<input value="CCTV에서 보낸 유해동물 대응 미션 현황"readonly="readonly" style="background-color: #864DD9; color: white; font-weight: 500; font-size:20px; border:none ;font-weight: bold; height: 36px; text-align: center; width: 520px; margin-top: -3px; margin-left: -3px"/>
               		<input id="numShow" value="사건 번호 000 대응 중" readonly="readonly" style="background-color: transparent ; font-weight: bold; border-color: transparent; color: dimgray; text-align: center; margin-left: -30px; width: 180px; margin-top: 10px; ">
               		<input id="zoneShow" value="E 구역에서" readonly="readonly" style="background-color: transparent ; font-weight: bold; border-color: transparent; color: dimgray; text-align: center; margin-left: 10px; width: 120px; margin-top: 10px">
               		<input id="animalShow" value="고라니 탐지됨" readonly="readonly" style="background-color: transparent ; font-weight: bold; border-color: transparent; color: dimgray; text-align: center; margin-left: 10px; width: 120px; margin-top: 10px">
              		
              		<img id="beginSign" src="${pageContext.request.contextPath}/resource/img/begin.png" style="width: 100px; height: 100px; margin-left: 30px; margin-top: 20px">
               		<img id="startSign" src="${pageContext.request.contextPath}/resource/img/arrived.png" style="width: 100px; height: 100px; margin-left: 60px; margin-top: 20px">
               		<img id="finishSign" src="${pageContext.request.contextPath}/resource/img/complete.png" style="width: 90px; height: 100px; margin-left: 50px; margin-top: 20px">
               		
               		<input id="beginText" value="A 구역 출동 중" readonly="readonly" style="background-color: transparent ; font-weight: bold; border-color: transparent; color: dimgray; text-align: center; margin-left: 10px; width: 150px">
               		<input id="startText" value="도착 및 처리 중" readonly="readonly" style="background-color: transparent ; font-weight: bold; border-color: transparent; color: dimgray; text-align: center; margin-left: 20px; width: 120px">
               		<input id="finishText" value="처리 완료" readonly="readonly" style="background-color: transparent ; font-weight: bold; border-color: transparent; color: dimgray; text-align: center; margin-left: 30px; width: 120px">
               </div>
       		</div>
       		
	       	<input value="유해동물 탐지 현황" readonly="readonly" style="background-color: #ADFF2F; ; color: black; margin-left:30px ;font-weight: 500; font-size:20px; border-color: transparent; font-weight: bold;position: absolute; margin-top:500px; height: 36px; padding: 0px; text-align: center; width: 1043px"/>
	       	
	       	<div style="border-color: transparent; margin-top: 545px; width: 1070px; padding-left: 5px">
		       <div class="container" style="background-color: #22252a;  margin-left: 25px; border-color: #ADFF2F; border-style:solid; border-width:medium; width: 1043px ">         
				  <table class="table hover" style="margin-bottom: 0px; margin-left: 0px">
				    <thead style="font-size: medium;">
				      <tr>
				        <th style="color: white; text-align: center; font-weight: bold; color: #ADFF2F;">탐지 주체</th>
				        <th style="color: white; text-align: center; font-weight: bold; color: #ADFF2F;">유해동물 이름</th>
				        <th style="color: white; text-align: center; font-weight: bold; color: #ADFF2F;">유해동물 등급</th>
				        <th style="color: white; text-align: center; font-weight: bold; color: #ADFF2F;">탐지 구역</th>
				      </tr>
				    </thead>
				    <tbody style="color: white; font-size:small;">
				      <tr id="jet1">
				        <td id="j1Col1" style="text-align: center; ">JetRacer 1</td>
				        <td><input id="j1Obj" value="*****  탐지대상 없음  *****" class="detectContent" readonly="readonly"></td>
				        <td><input id="j1Lev" value="*****  해당사항 없음  *****" class="detectContent" readonly="readonly"></td>
				        <td><input id="j1Loc" value="*****  해당사항 없음  *****" class="detectContent" readonly="readonly"></td>
				      </tr> 
				      <tr id="jet2">
				        <td id="j2Col1" style="text-align: center; ">JetRacer 2</td>
				        <td><input id="j2Obj" value="*****  탐지대상 없음  *****" class="detectContent" readonly="readonly"></td>
				        <td><input id="j2Lev" value="*****  해당사항 없음  *****" class="detectContent" readonly="readonly"></td>
				        <td><input id="j2Loc" value="*****  해당사항 없음  *****" class="detectContent" readonly="readonly"></td>
				      </tr>
				      <tr id="jet3">
				        <td id="j3Col1" style="text-align: center; ">JetRacer 3</td>
				        <td><input id="j3Obj" value="*****  탐지대상 없음  *****" class="detectContent" readonly="readonly"></td>
				        <td><input id="j3Lev" value="*****  해당사항 없음  *****" class="detectContent" readonly="readonly"></td>
				        <td><input id="j3Loc" value="*****  해당사항 없음  *****" class="detectContent" readonly="readonly"></td>
				      </tr>
				      <tr id="cctv1">
				        <td id="c1Col1" style="text-align: center; ">CCTV 1</td>
				        <td><input id="c1Obj" value="*****  탐지대상 없음  *****" class="detectContent" readonly="readonly"></td>
				        <td><input id="c1Lev" value="*****  해당사항 없음  *****" class="detectContent" readonly="readonly"></td>
				        <td><input id="c1Loc" value="*****  해당사항 없음  *****" class="detectContent" readonly="readonly"></td>
				      </tr>
				      <tr id="cctv2">
				        <td id="c2Col1" style="text-align: center; ">CCTV 2</td>
				        <td><input id="c2Obj" value="*****  탐지대상 없음  *****" class="detectContent" readonly="readonly"></td>
				        <td><input id="c2Lev" value="*****  해당사항 없음  *****" class="detectContent" readonly="readonly"></td>
				        <td><input id="c2Loc" value="*****  해당사항 없음  *****" class="detectContent" readonly="readonly"></td>
				      </tr>
				      <tr id="cctv3">
				        <td id="c3Col1" style="text-align: center;">CCTV 3</td>
						<td><input id="c3Obj" value="*****  탐지대상 없음  *****" class="detectContent" readonly="readonly"></td>
				        <td><input id="c3Lev" value="*****  해당사항 없음  *****" class="detectContent" readonly="readonly"></td>
				        <td><input id="c3Loc" value="*****  해당사항 없음  *****" class="detectContent" readonly="readonly"></td>
				      </tr>
				      <tr id="cctv4">
				        <td id="c4Col1" style="text-align: center;">CCTV 4</td>
				        <td><input id="c4Obj" value="*****  탐지대상 없음  *****" class="detectContent" readonly="readonly"></td>
				        <td><input id="c4Lev" value="*****  해당사항 없음  *****" class="detectContent" readonly="readonly"></td>
				        <td><input id="c4Loc" value="*****  해당사항 없음  *****" class="detectContent" readonly="readonly"></td>
				      </tr>
				    </tbody>
				  </table>
				</div>
	   		</div>
	   		
   		</div>
   	  </div>
    </body>
</html>