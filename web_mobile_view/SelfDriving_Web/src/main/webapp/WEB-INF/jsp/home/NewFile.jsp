<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<!-- bootswatch slate theme -->
      <link rel="stylesheet" href="${pageContext.request.contextPath}/resource/css/bootstrap.min.css">
<style>
canvas {
    border:1px solid #d3d3d3;
    background-color: #f1f1f1;
    position: absolute;
    top:0;
    bottom: 0;
    left: 0;
    right: 0;
    margin:auto;
}
</style>
</head>
<body onload="startGame()">

<script>
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
        this.canvas.width = window.innerHeight; // 캔버스 크기 성정
        this.canvas.height = window.innerHeight;
        this.scale = this.canvas.width / 500;
        this.context = this.canvas.getContext("2d"); // 캔버스에서 그리기 도구 객체 얻기
        this.context.scale(this.scale, this.scale);
        document.body.insertBefore(this.canvas, document.body.childNodes[0]); // 캔버스 태그를 body 태그 안에 넣기
        this.interval = setInterval(updateGameArea, 10); // 20ms마다 updateGameArea 함수를 실행
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
    this.moveAngle = 0; // 변화 각도
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
           this.moveLeft(); console.log("L1");}
       if(this.x <= 150 && this.y >= 50 && this.y <100) {
             this.turnLeft2(); console.log("L2");}
       if(this.x == 100 && this.y >= 100 && this.y <150) {
             this.moveDown(); console.log("L3");}
       if(this.y >= 150 && this.y <300 && this.angle >= Math.PI && this.angle <= Math.PI + 0.62 * Math.PI / 6) {
             this.moveDown2(); console.log("L4");}
       if(this.x == 50 && this.y >= 300 && this.y < 350) {
             this.moveDown(); console.log("L5");}
       if(this.x >= 50 && this.x < 100 && this.y >= 350 && this.angle >= Math.PI / 2 && this.angle <= Math.PI) {
             this.turnLeft3(); console.log("L6");}
       if(this.x >= 100 && this.x < 125 && this.y >= 400 && this.y < 425 && this.angle >= Math.PI / 2 && this.angle <= Math.PI) {
             this.turnRight6(); console.log("L7");}
       if(this.x >= 125 && this.x < 150 && this.y >= 425 && this.y < 450 && this.angle >= Math.PI / 2 && this.angle <= Math.PI) {
           this.turnLeft4(); console.log("L8");}
       if(this.x >= 150 && this.x < 400 && this.y == 450) {
           this.moveRight(); console.log("L9");}
       if(this.x >= 400 && this.angle > 0 && this.angle <= 3 * Math.PI / 2) {
           this.turnLeft5(); console.log("L10");}
       
       if(this.x == 450 && this.y > 100 && this.y <= 400) {
           this.moveUp1(); console.log("L11");}
       if(this.x > 400 && this.x <= 450 && this.y <= 100 && this.angle >= 2 * Math.PI / 2 && this.angle <= 5 * Math.PI / 2) { 
           this.turnLeft6(); console.log("L12");}
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
       this.moveAngle = 1.145;
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
        if(this.y <= 150) {this.x = 100; this.y = 150; return} //this.x = 100; this.y = 150; return
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
         if(this.y >= 300) {this.x = 50; this.y = 300; return} //this.x = 100; this.y = 150; return
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
    if(flag1 && direction1){car1.fixPosition1();} // 역방향 움직임
    if(flag2 && direction2){car2.fixPosition1();}
    if(flag3 && direction3){car3.fixPosition1();}
    
    if(flag1 && !direction1){car1.fixPosition2();} // 정방향 움직임
    if(flag2 && !direction2){car2.fixPosition2();}
    if(flag3 && !direction3){car3.fixPosition2();}
    
    car1.update(); // component 그리기
    car2.update();
    car3.update();

    flagA.update2();
    flagB.update2();
    console.log(car1.x, car1.y, car1.angle * 180 / Math.PI);
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

<p>Make sure the gamearea has focus, and use the arrow keys to move the red square around.</p>
<button onclick="changeFlag1()">red</button>
<button onclick="changeFlag2()">green</button>
<button onclick="changeFlag3()">blue</button>
<button onclick="changeDirection1()">red direction</button>
<button onclick="changeDirection2()">green direction</button>
<button onclick="changeDirection3()">blue direction</button>
</body>
</html>