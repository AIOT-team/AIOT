<%@ page language="java" contentType="text/html; charset=UTF-8"
	pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>No. 10</title>
<link rel="stylesheet"
	href="${pageContext.request.contextPath}/resource/bootstrap/css/bootstrap.min.css">
<script src="${pageContext.request.contextPath}/resource/jquery/jquery-3.4.1.min.js" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/paho-mqtt/1.0.1/mqttws31.min.js" type="text/javascript"></script>
<script
	src="${pageContext.request.contextPath}/resource/popper/popper.min.js"></script>
<script
	src="${pageContext.request.contextPath}/resource/bootstrap/js/bootstrap.min.js"></script>
<link rel="stylesheet"
	href="${pageContext.request.contextPath}/resource/jquery-ui/jquery-ui.min.css">
<script
	src="${pageContext.request.contextPath}/resource/jquery-ui/jquery-ui.min.js"></script>

<script>
var ipid
$(function(){
	ipid =new Date().getTime().toString()
	client = new Paho.MQTT.Client("192.168.3.184", 61614, ipid);
	client.onMessageArrived = onMessageArrived;
	client.connect({onSuccess:onConnect});
});
function onConnect() {
	client.subscribe("/req/2cctv");
	console.log("연결됐다고 알림!");
}
 $(document).ready(function() {
    setInterval(getinterval, 300);
});   
	var bat=0;
    var lastSendtime=Date.now();
 function getinterval(){
	interval= Date.now()-lastSendtime;
		if(interval>1000){
			console.log("연결이 끊긴다음 몇초가 흘렀는지를 보여주는 console.log의 시간:"+interval);
			response();
		}
} 
 function response(){
	console.log("답장을 보내요.")
	  message = new Paho.MQTT.Message(ipid);
	  message.destinationName = "/res/2cctv";
	  client.send(message);
}
 var box;
function onMessageArrived(message) {
	 console.log("hioyng")
	if(message.destinationName =="/req/2cctv") {
		console.log("hioyng")
		response();
		lastSendtime=Date.now();
		const json = message.payloadString;
		const obj = JSON.parse(json);
		txtcanvas2=obj.Class;
		
		bat=obj.Battery;
		
		//console.log(bat);
		if(obj.Box)
			{
			console.log(obj.Box);
			box=obj.Box;			
			obj.Box.forEach(function (item, index, array) {
				  boxx=(item[0]+item[2])*1200/640
				  boxy=(item[1]+item[3])*900/480;
				  arr_boxx.push(boxx);
				  arr_boxy.push(boxy);
				})
			//console.log(obj.Box[0][0]);
			//boxx=obj.Box[0]*1200/320;
			//boxy=obj.BoxMiddle[1]*900/240;
			}
		image.src="data:image/jpg;base64,"+ obj.Cam;
	}
}
var arr_boxx=[];
var arr_boxy=[];
$(function() {
	//1200="1200px"
	//900="900px"
	canvas1stfloor =document.createElement("canvas")
	canvas1stfloor.width = 1200;
	canvas1stfloor.height = 900;
	ctx1stfloor = canvas1stfloor.getContext("2d");
	document.body.appendChild(canvas1stfloor);
});
var image = new Image();
image.onload = function() {
  	ctx1stfloor.drawImage(image, 0, 0,1200,900);
};
image.src = "";
$(function() {
	canvastrapezoid =document.createElement("canvas")
	canvastrapezoid.width = 1200;
	canvastrapezoid.height = 900;
	canvastrapezoid.style.position= "absolute";
	canvastrapezoid.style.bottom="0";
	ctxtrapezoid = canvastrapezoid.getContext("2d");
	var grd = ctxtrapezoid.createLinearGradient(0, canvastrapezoid.height, 0, 0);
	grd.addColorStop(0, "rgba(5,229,238,1)");
	grd.addColorStop(0.1,"rgba(5,229,238,0)");
	grd.addColorStop(0.2,"rgba(5,229,238,1)");
	grd.addColorStop(0.3,"rgba(5,229,238,0)");
	grd.addColorStop(0.4,"rgba(5,229,238,1)");
	grd.addColorStop(0.5,"rgba(5,229,238,0)");
	grd.addColorStop(0.6,"rgba(5,229,238,1)");
	grd.addColorStop(1, "rgba(5,229,238,0)");
	document.body.insertBefore(canvastrapezoid, document.body.childNodes[0]);
	setInterval(drawtrapezoid, 200);
	function drawtrapezoid(){
		ctxtrapezoid.clearRect(0,0,canvastrapezoid.width, canvastrapezoid.height);
		if(box){
	
	ctxtrapezoid.moveTo(box[0][0]*1200/320, box[0][3]*900/240);
	ctxtrapezoid.lineTo(box[0][2]*1200/320, box[0][3]*900/240);
	//ctxtrapezoid.moveTo(canvastrapezoid.width/2-70, canvastrapezoid.height-200);
	//ctxtrapezoid.lineTo(canvastrapezoid.width/2-130, canvastrapezoid.height-200);
	ctxtrapezoid.lineWidth = "2";
	ctxtrapezoid.lineTo(canvastrapezoid.width, canvastrapezoid.height);
	ctxtrapezoid.lineTo(0, canvastrapezoid.height);
	ctxtrapezoid.fillStyle = grd;
	ctxtrapezoid.fill();
	
	ctxtrapezoid.clearRect(0,0,canvastrapezoid.width, box[0][3]*900/240);
	
	ctxtrapezoid.beginPath();
	ctxtrapezoid.strokeStyle="rgb(5,229,238)";
	ctxtrapezoid.moveTo(box[0][0]*1200/320, canvastrapezoid.height/2);
	ctxtrapezoid.lineTo(box[0][2]*1200/320, canvastrapezoid.height/2);
	ctxtrapezoid.lineTo(canvastrapezoid.width, canvastrapezoid.height);
	ctxtrapezoid.lineTo(0, canvastrapezoid.height);
	ctxtrapezoid.lineTo(box[0][0]*1200/320, canvastrapezoid.height/2);
	ctxtrapezoid.stroke();
	box= null;
		}
		else{
			ctxtrapezoid.beginPath();
			ctxtrapezoid.strokeStyle="rgb(5,229,238)";
			ctxtrapezoid.moveTo(canvastrapezoid.width/2-70, canvastrapezoid.height/2);
			ctxtrapezoid.lineTo(canvastrapezoid.width/2+70, canvastrapezoid.height/2);
			ctxtrapezoid.lineTo(canvastrapezoid.width, canvastrapezoid.height);
			ctxtrapezoid.lineTo(0, canvastrapezoid.height);
			ctxtrapezoid.lineTo(box[0][0]*1200/320, canvastrapezoid.height/2);
			ctxtrapezoid.stroke();
		}
	}
});
$(function() {
	canvas2ndfloor =document.createElement("canvas")
	canvas2ndfloor.width = 1200;
	canvas2ndfloor.height = 900;
	canvas2ndfloor.style.position= "absolute";
	
	ctx2ndfloor = canvas2ndfloor.getContext("2d");
	ctx2ndfloor.font = "40px Arial";
	setInterval(setText, 20);
	document.body.insertBefore(canvas2ndfloor, document.body.childNodes[0]);
});
$(function() {
	canvas3rdfloor =document.createElement("canvas")
	canvas3rdfloor.width = 1200;
	canvas3rdfloor.height = 900;
	canvas3rdfloor.style.position= "absolute";
	canvas3rdfloor.style.right="0"
	canvas3rdfloor.style.bottom="0";
	ctx3rdfloor = canvas3rdfloor.getContext("2d");
	ctx3rdfloor.font = "50px Arial";

	ctx3rdfloor.textAlign = "center";
	ctx3rdfloor.fillText("속도km/h",canvas3rdfloor.width/2, canvas3rdfloor.height/2);
	document.body.insertBefore(canvas3rdfloor, document.body.childNodes[0]);
});

$(function() {
	canvasprogress =document.createElement("canvas")
	canvasprogress.style.position= "absolute";
	canvasprogress.style.left = "100px";
	canvasprogress.style.bottom="0"
  //var can = document.getElementById('canvas'),
      spanProcent = document.createElement("span"),
      spanProcent.style.position= "absolute";
		 spanProcent.style.right="0"
      ctxprogress = canvasprogress.getContext("2d");
	  
 
  var posX = canvasprogress.width / 2,
      posY = canvasprogress.height / 2,
      procent = 0,
      oneProcent = 360 / 100;
  result = oneProcent * bat;
  ctxprogress.lineCap = 'round';
  document.body.insertBefore(canvasprogress, document.body.childNodes[0]);
  //document.body.insertBefore(spanProcent, document.body.childNodes[0]);
  arcMove();
  
  function arcMove(){
	  
    var barcolor = 'black';
    var acrInterval = setInterval (function() {
    	result = oneProcent * bat;
    	console.log("hi")
      ctxprogress.clearRect( 0, 0, canvasprogress.width, canvasprogress.height );
      ctxprogress.font = "50px Georgia";
      if(Math.round(result/360*100)>80){
    	  barcolor="rgba(30, 130, 76, 0.8)"
      }else if(Math.round(result/360*100)>60){
    	  barcolor="rgba(247, 202, 24, 0.8)"
      }else{
    	  barcolor="rgba(242, 38, 19, 0.8)"
      }
      ctxprogress.fillStyle= barcolor;
      ctxprogress.fillText(Math.round(result/360*100), canvasprogress.width/2-10, canvasprogress.height/2+10);
      
      ctxprogress.beginPath();
      ctxprogress.arc( posX, posY, 60, (Math.PI/180) * 90, (Math.PI/180) * (90 + 360) );
      ctxprogress.strokeStyle = '#b1b1b1';
      ctxprogress.lineWidth = '20';
      ctxprogress.stroke();

      ctxprogress.beginPath();
      ctxprogress.strokeStyle = barcolor;
      ctxprogress.lineWidth = '20';
      ctxprogress.arc( posX, posY, 60, (Math.PI/180) * 90, (Math.PI/180) * (90 + result) );
      ctxprogress.stroke();
    }, 1000);
    
  }
});

$(function() {
	canvasprogress2 =document.createElement("canvas")
	canvasprogress2.style.position= "absolute";
	canvasprogress2.style.bottom="0";
	canvasprogress2.width=1200/2;
	canvasprogress2.height=canvasprogress2.width;
	nullbi=1200/4;
	canvasprogress2.style.left=nullbi+"px";
      spanProcent2 = document.createElement("span"),
      spanProcent2.style.position= "absolute";
		 spanProcent2.style.right="0"
      ctxprogress2 = canvasprogress2.getContext("2d");
	  
 
  var posX2 = canvasprogress2.width / 2,
      posY2 = canvasprogress2.height / 2,
      procent2 = 0,
      oneProcent2 = 360 / 100;
  result2 = oneProcent2 * bat;
  document.body.insertBefore(canvasprogress2, document.body.childNodes[0]);
  //document.body.insertBefore(spanProcent, document.body.childNodes[0]);
  arcMove2();
  
  function arcMove2(){
	  
    var barcolor2 = 'black';
    var acrInterval2 = setInterval (function() {
    	result2 = oneProcent2 * bat;
    	console.log("hi")
      ctxprogress2.clearRect( 0, 0, canvasprogress2.width, canvasprogress2.height );
      ctxprogress2.font = "80px Georgia";
      
	  
     	if(Math.round(result/360*100)<60){
    	  barcolor2="black"
      }else{
    	  barcolor2="rgba(242, 38, 19, 0.8)"
      }
      ctxprogress2.fillStyle= barcolor2
      ctxprogress2.fillText(Math.round(result/360*100), canvasprogress2.width/2-10, canvasprogress2.height-30);
      ctxprogress2.beginPath();
      ctxprogress2.arc( posX2, 900, canvasprogress2.height/4, (Math.PI/180) * 180, (Math.PI/180) * (90 + 270) );
      ctxprogress2.strokeStyle = '#b1b1b1';
      ctxprogress2.lineWidth = '20';
      ctxprogress2.stroke();
	
      ctxprogress2.beginPath();
      ctxprogress2.strokeStyle = barcolor2;
      ctxprogress2.lineWidth = '20';
      ctxprogress2.setLineDash([4, 4]);
      ctxprogress2.arc( posX2, 900, canvasprogress2.height/4, (Math.PI/180) * 180, (Math.PI/180) * (180+Math.round(result/180*100)) );
      ctxprogress2.stroke();
    }, 1000);
    
  }
});

$(function() {
	canvascircle =document.createElement("canvas")
	canvascircle.width = 1200;
	canvascircle.height = 900;
	canvascircle.style.position= "absolute";
	
	ctxcirclefloor = canvascircle.getContext("2d");
	
	//ctxcirclefloor.stroke();
	setInterval(drawCircle,1);
	document.body.insertBefore(canvascircle, document.body.childNodes[0]);
});
var count=0;
function drawCircle(){
	ctxcirclefloor.clearRect(0,0,1200,900);
	count++;
	ctxcirclefloor.beginPath();
	ctxcirclefloor.lineWidth="5"
	ctxcirclefloor.strokeStyle = "#20d2f4"
	if(arr_boxx != []){
		arr_boxx.forEach(function(element){
			console.log("그린다")
		boxx=element;
		boxy=arr_boxy.pop();
		//boxx=arr_boxx.pop()
		//boxy=arr_boxy.pop()
	
	if(count%2==0){
		ctxcirclefloor.setLineDash([4, 4])
	}
	else{
		ctxcirclefloor.setLineDash([0,4,4])
	}
		ctxcirclefloor.arc(boxx,boxy, 100, 0, 2 * Math.PI);
		ctxcirclefloor.moveTo(boxx+125,boxy)
		ctxcirclefloor.arc(boxx,boxy, 125, 0, 2 * Math.PI);
		ctxcirclefloor.closePath();
		ctxcirclefloor.stroke();
		
		});
		arr_boxx.length=0;
		arr_boxy.length=0;
	}
		
	}
function setText(){
	ctx2ndfloor.clearRect(0,0,1200,900);
	ctx2ndfloor.fillText(txtcanvas2, 30, 100);
}
var txtcanvas2;
/* 
$(function() {
	canvas3rdfloor =document.createElement("canvas")
	canvas3rdfloor.width = 1200/4;
	canvas3rdfloor.height = 900/4;
	canvas3rdfloor.style.position= "absolute";
	
	ctx3rdfloor = canvas2ndfloor.getContext("2d");
	ctx3rdfloor.font = "30px Arial";
	ctx3rdfloor.fillText("Hello World", 100, 100);
	document.body.insertBefore(canvas3rdfloor, document.body.childNodes[0]);
}); */
function startGame() {
    car1 = new component(15, 20, "red", 50, 350, Math.PI); // component 생성
    car2 = new component(15, 20, "green", 300, 50, Math.PI / 2);
    car3 = new component(15, 20, "blue", 350, 50, Math.PI / 2);
    flagA = new component(15, 20, "A", 300, 50, Math.PI / 2);
    flagB = new component(15, 20, "B", 350, 50, Math.PI / 2);
    myGameArea.start(); 
}

function stop() { // 테스트 용
   clearInterval(myGameArea.interval);
}

//var scale = myGameArea.canvas.width / 500;

var myGameArea = {
    canvas : document.createElement("canvas"), // 캔버스 태그 생성
    start : function() {
        this.canvas.width = 900/3; // 캔버스 크기 성정
        this.canvas.height = 900/3;
this.canvas.style.position = "absolute";
        this.canvas.style.right= "0";
        this.canvas.style.bottom= "0";
        this.scale = this.canvas.width / 500;
        this.context = this.canvas.getContext("2d"); // 캔버스에서 그리기 도구 객체 얻기
        this.context.scale(this.scale, this.scale);
        document.body.insertBefore(this.canvas, document.body.childNodes[0]); // 캔버스 태그를 body 태그 안에 넣기
        this.interval = setInterval(updateGameArea, 20); // 20ms마다 updateGameArea 함수를 실행
    },
    stop : function() {
        clearInterval(this.interval);
    },    
    clear : function() {
        this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
}

var flag1 = false;
var flag2 = false;
var flag3 = false;

var direction1 = false;
var direction2 = false;
var direction3 = false;

function changeFlag1() {
   if(flag1 == false) flag1 = true;
   else flag1 = false;
   console.log(flag1);
}

function changeFlag2() {
   if(flag2 == false) flag2 = true;
   else flag2 = false;
   console.log(flag2);
}

function changeFlag3() {
   if(flag3 == false) flag3 = true;
   else flag3 = false;
   console.log(flag3);
}

function changeDirection1() {
   if(direction1 == false) direction1 = true;
   else direction1 = false;
   console.log(direction1);
}

function changeDirection2() {
   if(direction2 == false) direction2 = true;
   else direction2 = false;
   console.log(direction2);
}

function changeDirection3() {
   if(direction3 == false) direction3 = true;
   else direction3 = false;
   console.log(direction3);
}


function component(width, height, color, x, y, angle) {
    this.width = width;
    this.height = height;
    this.speed = 0;
    this.angle = angle; // 현대 각도
    this.moveAngle = bat/90* 0; // 변화 각도
    this.x = x; // component 좌표
    this.y = y;
    this.update = function() {
           ctx = myGameArea.context; // 그리기 객체 도구 얻기
           
           // component rotation에 필요
           ctx.save();
           ctx.translate(this.x, this.y);
           ctx.rotate(this.angle);
           
           ctx.fillStyle = color; // 직사각형 색 설정
           ctx.fillRect(this.width / -2, this.height / -2, this.width, this.height); // 직사각형 그리기
           
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
			ctx.fillRect(this.width / -2, this.height / -2-20, 30, 20);
            ctx.restore();
    }
    
    this.fixPosition1 = function() {
       if(this.x >= 150 && this.x < 400 && this.y == 50) {
           this.moveRight(); console.log("1");}
        if(this.x >= 400 && this.angle >= Math.PI / 2 && this.angle < Math.PI) { 
           this.turnRight1(); console.log("2");}
        if(this.x == 450 && this.y >= 100 && this.y < 400) {
           this.moveDown(); console.log("3");}
        if(this.y >= 400 && this.angle >= Math.PI / 2 && this.angle < 3 * Math.PI / 2) {
           this.turnRight2(); console.log("4");}
        if(this.x > 160 && this.x <= 400 && this.y == 450) {
           this.moveLeft(); console.log("5");}
        
        if(this.x > 135 && this.x <= 160 && this.angle >= 3 * Math.PI / 2 && this.angle < 2 * Math.PI) {
           this.turnRight3(); console.log("6");}
          if(this.x > 100 && this.x <= 135 && this.angle >= 3 * Math.PI / 2 && this.angle < 2 * Math.PI) {
             this.turnLeft(); console.log("7");}
          
          if(this.x > 50 && this.x <= 100 && this.y >= 350 && this.angle >= 3 * Math.PI / 2 && this.angle <= 2 * Math.PI) {
             this.turnRight4(); console.log("8");}
          
          if(this.x == 50 && this.y > 300 && this.y <= 350) {
             this.moveUp1(); console.log("9");}
          if(this.y > 150 && this.y <=300 && this.angle >= 2 * Math.PI && this.angle <= 12.62 * Math.PI / 6) {
             this.moveUp2(); console.log("10");}
          
          if(this.x == 100 && this.y > 100 && this.y <=150) {
             this.moveUp1(); console.log("11");}
          if(this.x >= 100 && this.y > 50 && this.y <=100  && this.angle >= 2 * Math.PI && this.angle < 5 * Math.PI / 2) {
             this.turnRight5(); console.log("12");}
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
       this.speed = bat/90;
        this.angle = Math.PI / 2;
        this.x += this.speed * Math.sin(this.angle);
        this.y -= this.speed * Math.cos(this.angle);
    }
    
    this.turnRight1 = function() {
       this.speed = bat/90;
       this.moveAngle = bat/90* 1.145;
        this.angle +=  this.moveAngle * Math.PI / 180;
        if (this.angle >= Math.PI) { this.x = 450; this.y = 100; return;}
        this.x += this.speed * Math.sin(this.angle);
        this.y -= this.speed * Math.cos(this.angle);
    }
    
    this.moveDown = function() {
       this.speed = bat/90;
        this.angle = Math.PI;
        this.x += this.speed * Math.sin(this.angle);
        this.y -= this.speed * Math.cos(this.angle);
    }
    
    this.turnRight2 = function() {
       this.speed = bat/90;
       this.moveAngle = bat/90* 1.145;
        this.angle +=  this.moveAngle * Math.PI / 180;
        if (this.angle >= 3 * Math.PI / 2) { this.x = 400; this.y = 450; return;}
        this.x += this.speed * Math.sin(this.angle);
        this.y -= this.speed * Math.cos(this.angle);
    }
    
    this.moveLeft = function() {
       this.speed = bat/90;
        this.angle = 3 * Math.PI / 2;
        this.x += this.speed * Math.sin(this.angle);
        this.y -= this.speed * Math.cos(this.angle);
    }
    
    this.turnRight3 = function() {
       this.speed = bat/90;
       this.moveAngle = bat/90* 2;
        this.angle +=  this.moveAngle * Math.PI / 180;
        if (this.angle >= 11 * Math.PI / 6) {this.x = 135; this.y = 435; return;} //this.x = 135; this.y = 435; return;
        this.x += this.speed * Math.sin(this.angle);
        this.y -= this.speed * Math.cos(this.angle);
    }
    
    this.turnLeft = function() {
       this.speed = bat/90;
       this.moveAngle = bat/90* (-0.7);
        this.angle +=  this.moveAngle * Math.PI / 180;
        if (this.x < 101) {this.x = 100; this.y = 400; this.angle = 3 * Math.PI / 2; return;} // this.x = 100; this.y = 400; return
        this.x += this.speed * Math.sin(this.angle);
        this.y -= this.speed * Math.cos(this.angle);
    }
    
    this.turnRight4 = function() {
       this.speed = bat/90;
       this.moveAngle = bat/90* 1.145;
        this.angle +=  this.moveAngle * Math.PI / 180;
        if (this.angle >= 2 * Math.PI) {this.x = 50; this.y = 350; return;}
        this.x += this.speed * Math.sin(this.angle);
        this.y -= this.speed * Math.cos(this.angle);
    }
    
    this.moveUp1 = function() {
       this.speed = bat/90;
        this.angle =  2 * Math.PI;
        this.x += this.speed * Math.sin(this.angle);
        this.y -= this.speed * Math.cos(this.angle);
    }
    
    this.moveUp2 = function() {
       this.speed = bat/90;
        this.angle = 12.62 * Math.PI / 6;
        this.x += this.speed * Math.sin(this.angle);
        this.y -= this.speed * Math.cos(this.angle);
        if(this.y <= 150) {this.x = 100; this.y = 150; return} //this.x = 100; this.y = 150; return
    }
    
    this.turnRight5 = function() {
       this.speed = bat/90;
       this.moveAngle = bat/90*1.145;
        this.angle +=  this.moveAngle * Math.PI / 180;
        if (this.angle >= 5 * Math.PI / 2) { this.x = 150; this.y = 50; return}
        this.x += this.speed * Math.sin(this.angle);
        this.y -= this.speed * Math.cos(this.angle);
    }
    
    this.turnLeft2 = function() {
        this.speed = bat/90;
        this.moveAngle = bat/90*(-1.145);
         this.angle +=  this.moveAngle * Math.PI / 180;
         if (this.angle <= Math.PI) { this.x = 100; this.y = 100; return}
         this.x += this.speed * Math.sin(this.angle);
         this.y -= this.speed * Math.cos(this.angle);
     }
    
    this.moveDown2 = function() {
        this.speed = bat/90;
         this.angle = Math.PI + 0.62 * Math.PI / 6;
         this.x += this.speed * Math.sin(this.angle);
         this.y -= this.speed * Math.cos(this.angle);
         if(this.y >= 300) {this.x = 50; this.y = 300; return} //this.x = 100; this.y = 150; return
     }
    
    this.turnLeft3 = function() {
        this.speed = bat/90;
        this.moveAngle = bat/90* (-1.145);
         this.angle +=  this.moveAngle * Math.PI / 180;
         if (this.angle <= Math.PI / 2) {this.x = 100; this.y = 400; this.angle = Math.PI / 2; return;}
         this.x += this.speed * Math.sin(this.angle);
         this.y -= this.speed * Math.cos(this.angle);
     }
    
    this.turnRight6 = function() {
        this.speed = bat/90;
        this.moveAngle = bat/90*2.25;
         this.angle +=  this.moveAngle * Math.PI / 180;
         if (this.angle >= Math.PI) {this.x = 125; this.y = 425; this.angle = Math.PI; return;} // this.x = 100; this.y = 400; return
         this.x += this.speed * Math.sin(this.angle);
         this.y -= this.speed * Math.cos(this.angle);
     }
    
    this.turnLeft4 = function() {
       this.speed = bat/90;
       this.moveAngle = bat/90*(-2.25);
        this.angle +=  this.moveAngle * Math.PI / 180;
        if (this.angle <= Math.PI / 2) {this.x = 150; this.y = 450; this.angle = Math.PI / 2; return;} //this.x = 135; this.y = 435; return;
        this.x += this.speed * Math.sin(this.angle);
        this.y -= this.speed * Math.cos(this.angle);
    }
    
    this.turnLeft5 = function() {
        this.speed = bat/90;
        this.moveAngle = bat/90*(-1.145);
         this.angle +=  this.moveAngle * Math.PI / 180;
         if (this.angle <= 0) {this.x = 450; this.y = 400; this.angle = 0; return;}
         this.x += this.speed * Math.sin(this.angle);
         this.y -= this.speed * Math.cos(this.angle);
     }
    
    this.turnLeft6 = function() {
        this.speed = bat/90;
        this.moveAngle = bat/90*(-1.145);
         this.angle +=  this.moveAngle * Math.PI / 180;
         if (this.angle <= 3 * Math.PI / 2) {this.x = 400; this.y = 50; this.angle = 3 * Math.PI / 2; return;}
         this.x += this.speed * Math.sin(this.angle);
         this.y -= this.speed * Math.cos(this.angle);
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

    car1.fixPosition2(); // 정방향 움직임
    car2.fixPosition2();
   car3.fixPosition2();
    
    car1.update(); // component 그리기
    car2.update();
    car3.update();

    flagA.update2();
    flagB.update2();
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
</body>
</html>