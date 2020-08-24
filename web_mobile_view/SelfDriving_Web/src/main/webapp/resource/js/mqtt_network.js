
			$(function(){
			client = new Paho.MQTT.Client(location.hostname, 61614, new Date().getTime().toString());
			client.onMessageArrived = onMessageArrived;
			client.connect({onSuccess:onConnect});
			});
	
			function onConnect() {
				//console.log("mqtt broker connected")
				client.subscribe("/1cctv");
				client.subscribe("/2cctv");
				client.subscribe("/3cctv");
				client.subscribe("/4cctv");
				client.subscribe("/sensor");
			}
			
			function onMessageArrived(message) {
				if(message.destinationName =="/1cctv") {
					$("#cameraView1").attr("src", "data:image/jpg;base64,"+ message.payloadString);
				}
				if(message.destinationName =="/2cctv") {
					$("#cameraView2").attr("src", "data:image/jpg;base64,"+ message.payloadString);
				}
				if(message.destinationName =="/3cctv") {
					$("#cameraView3").attr("src", "data:image/jpg;base64,"+ message.payloadString);
				}
				if(message.destinationName =="/4cctv") {
					const topic = message.destinationName;
					//console.log(typeof(topic));
					
					const json = message.payloadString;
					const obj = JSON.parse(json);

					$("#cameraView4").attr("src", "data:image/jpg;base64,"+ obj.Cam);
					//이미지에 탐지된 클래스에 대한 정보
					//console.log(obj.Class)
					
					var jsonTopic = JSON.stringify(topic);
					var jsonData = JSON.stringify(obj);
					//console.log(typeof(jsonData));
					
					if ( obj.Class.length != 0){
						$.ajax({
							type:"POST",
							url:"${pageContext.request.contextPath}/animal/saveImage.do",
							contentType: "application/json;charset=UTF-8",
							dataType: "json",
							data: jsonData
						});
					}
				}
				if(message.destinationName =="/sensor") {
					const json = message.payloadString;
					const obj = JSON.parse(json);
					$("#Battery").attr("value", obj.Battery);
				}
			}