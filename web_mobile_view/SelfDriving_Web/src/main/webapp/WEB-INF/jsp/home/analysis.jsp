<%@ page contentType="text/html; charset=UTF-8"%>
<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core"%>
<%@ taglib prefix="fmt" uri="http://java.sun.com/jsp/jstl/fmt"%>

<!DOCTYPE html>
<html>
  <head> 
	    <meta charset="utf-8">
		<meta http-equiv="X-UA-Compatible" content="IE=edge">
	    <title>AIOT FINAL PROJECT | TEAM 2</title>
	    <meta name="description" content="">
	    <meta name="viewport" content="width=device-width, initial-scale=1">
	    <meta name="robots" content="all,follow">
	    
	    <script src="${pageContext.request.contextPath}/resource/jquery/jquery-3.4.1.min.js" crossorigin="anonymous"></script>
	    <link rel="stylesheet" href="${pageContext.request.contextPath}/resource/jquery-ui/jquery-ui.min.css">
	    
	    <link rel="stylesheet" href="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/bootstrap/css/bootstrap.min.css">
	    <link rel="stylesheet" href="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/font-awesome/css/font-awesome.min.css">
	    <link rel="stylesheet" href="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/css/font.css">
	    <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Muli:300,400,700">
	    <link rel="stylesheet" href="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/css/style.default.css" id="theme-stylesheet">
		<link rel=icon href="${pageContext.request.contextPath}/resource/img/jetracer.png">
		
		<script src="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/bootstrap/js/bootstrap.min.js"></script>
		<script src="${pageContext.request.contextPath}/resource/popper/popper.min.js"></script>
		<script src="${pageContext.request.contextPath}/resource/jquery-ui/jquery-ui.min.js"></script>
		<script src="https://cdnjs.cloudflare.com/ajax/libs/paho-mqtt/1.0.1/mqttws31.min.js" type="text/javascript"></script>
		<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
		
		<script src="https://code.highcharts.com/highcharts.js"></script>
		<script src="https://code.highcharts.com/highcharts-more.js"></script>
		<script src="https://code.highcharts.com/modules/solid-gauge.js"></script>
		<script src="https://code.highcharts.com/modules/exporting.js"></script>
		<script src="https://code.highcharts.com/modules/export-data.js"></script>
		<script src="https://code.highcharts.com/modules/accessibility.js"></script>
		<script src="https://code.highcharts.com/modules/series-label.js"></script>
		<script src="<%=application.getContextPath()%>/resources/highcharts/code/themes/gray.js"></script>

	    <script src="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/popper.js/umd/popper.min.js"> </script>
	    <script src="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/bootstrap/js/bootstrap.min.js"></script>
	    <script src="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/jquery.cookie/jquery.cookie.js"> </script>
	    <script src="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/jquery-validation/jquery.validate.min.js"></script>
	    <script src="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/js/front.js"></script>
	</head>
	
	<body onload="chart();chart2();chart3()">
	
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
        <span class="heading" style="color: #DB6574">CATEGORIES</span>
        <ul class="list-unstyled">
          <li><a href="${pageContext.request.contextPath}/home/main.do" style="color: lightgray"> <i class="icon-home"></i>메인 페이지</a></li>
	          <li><a href="${pageContext.request.contextPath}/home/jetracer.do" style="color: lightgray"> <i class="icon-writing-whiteboard"></i>탐지봇 현황 </a></li>
	          <li><a href="${pageContext.request.contextPath}/home/history.do" style="color: lightgray"> <i class="icon-grid"></i>탐지 히스토리 조회 </a></li>
	          <li><a href="${pageContext.request.contextPath}/home/status.do" style="color: lightgray"> <i class="icon-padnote"></i>실시간 탐지 | 대응 현황</a></li>
	      	  <li class="active"><a href="${pageContext.request.contextPath}/home/analysis.do" style="color: lightgray"> <i class="icon-chart"></i>탐지 결과 분석  </a></li>
        </ul>
      </nav>    
	      <div class="page-content" style="top: -50px; height: 1080px; padding-bottom: 0px; " >

			<div style = "width:97%;  height:300px;">
			<div id="container"style="margin-top: 50px">
	      	<select id="term1" style="margin-left: 100px">
		    	<option value="hour">시별</option>
		    	<option value="day">일별</option>
		    	<option value="month">월별</option>
		    	<option value="year">연도별</option>
			</select>
			</div>
	      	<div id="containerc" class="chart_container" style="width:100%; float:left; height:280px; padding-top: 40px; padding-left: 100px;">
	      	DB에 연결하지 못했습니다.</div>
	      	<div id="container">
	      	<select id="term2" style="margin-left: 100px">
		    	<option value="all">전체기간</option>
		    	<option value="oneweek">최근일주일</option>
		    	<option value="onemonth">최근한달</option>
		    	<option value="oneyear">최근일년</option>
				</select>
			</div>
	      	<div id="container2" class="chart_container" style="width:100%; float:left; height:280px; padding-top: 40px; padding-left: 100px;">
	      	DB에 연결하지 못했습니다.</div>
	      	<div id="container">
	      	<select id="term3" style="margin-left: 100px">
		    	<option value="all">전체기간</option>
		    	<option value="oneweek">최근일주일</option>
		    	<option value="onemonth">최근한달</option>
		    	<option value="oneyear">최근일년</option>
			</select>
		  	</div>
	      	<div id="container3" class="chart_container" style="width:100%; float:left; height:280px; padding-top: 40px; padding-left: 100px;">
	      	DB에 연결하지 못했습니다.</div>
	      	</div>
	      	</div>
	      <script>
	      var numbers2;
	      var numbers3;
	      var numbers4;
	      var numbers5;
	      var numbers6;
	      var numbers7;
	      var maxdactno;
	      var maxdactnocctv;
	      var timearr=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23'];
	      var timedactnoarr=[0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0]
	      var termval;
	      var termtranslate={"all":"전체기간","oneweek":"최근 일주일","onemonth":"최근 한달","oneyear":"최근 일년",};
	      var pertranslate={"hour":"시별","day":"일별","month":"월별","year":"연도별",};
	      
	      $(function(){
				client = new Paho.MQTT.Client(location.hostname, 61614, new Date().getTime().toString());
				client.onMessageArrived = onMessageArrived;
				client.connect({onSuccess:onConnect});
			});
			
			function onConnect() {
				console.log("mqtt broker connected")
				client.subscribe("/mirror");
			}
			function onMessageArrived(message) {
					if(message.destinationName =="/mirror") {
					const json = message.payloadString;
					const obj = JSON.parse(json);
					//$("#mirrorView").attr("src", "data:image/jpg;base64,"+ obj.Cam);
					if(obj.direction=="left"){
						location.href="status.do";
					}else if (obj.direction=="right"){
						alert("최하단페이지입니다.");
					}
				}
}
	      $('#term1').change(function(){
	    	  $.ajax({
	    		type: "POST",
	    		url: "${pageContext.request.contextPath}/home/analysisMonthwithterm.do",
	    	  	data: $(this).val(),
	    	  	async: false,
	    	  	success: function(monthlist){
					//monthlist.filter(myFunction);
					 numbers2 = monthlist.map(myFunction);
					 numbers3 = monthlist.map(myFunction2);
					 console.log(numbers2);
					 console.log(numbers3);
					 maxdactnocctv=numbers4.indexOf(Math.max(...numbers4));
					chart();
				}
	    	  }) // $.ajax
	      });
	      $('#term2').change(function(){
	    	  $.ajax({
	    		type: "POST",
	    	  	url:'${pageContext.request.contextPath}/home/analysisRegionwithterm.do',
	    	  	data: $(this).val(),
	    	  	async: false,
	    	  	success:function(regionlist){
 					//monthlist.filter(myFunction);
					 numbers4 = regionlist.map(myFunction4);
					 numbers5 = regionlist.map(myFunction3);
					maxdactnocctv=numbers4.indexOf(Math.max(...numbers4));
					chart2();
				}
	    	  }) // $.ajax
	      });
	      $('#term3').change(function(){
	    	  $.ajax({
	    		type: "POST",
	    	  	url:'${pageContext.request.contextPath}/home/analysisHourwithterm.do',
	    	  	data: $(this).val(),
	    	  	//data: JSON.stringify({"term":termval}),
	    	  	//dataType: "json",
	    	  	async: false,
	    	  	success:function(hourlist){
	    	  		numbers6 = hourlist.map(myFunction5);
					 numbers7 = hourlist.map(myFunction6);
					 console.log(numbers6);
					 console.log(numbers7);
					 maxdactno=numbers6.indexOf(Math.max(...numbers6));
					 chart3();
	    	  	}
	    	  }) // $.ajax
	      });
	    	  $.ajax({
	 				type: "POST",
	 				async: false,
	 				url: "${pageContext.request.contextPath}/home/analysisMonth.do",
	 				success: 
		   			function(monthlist){
	 					//monthlist.filter(myFunction);
	 					 numbers2 = monthlist.map(myFunction);
	 					 numbers3 = monthlist.map(myFunction2);
	 				}
	      		});
	      		$.ajax({
					type: "POST",
					async: false,
					url: "${pageContext.request.contextPath}/home/analysisHour.do",
					success: 
		   			function(hourlist){
						//console.log(hourlist);
						 numbers6 = hourlist.map(myFunction5);
						 numbers7 = hourlist.map(myFunction6);
						 console.log(numbers6);
						 console.log(numbers7);
						 maxdactno=numbers6.indexOf(Math.max(...numbers6));
						 
					}
	    		});
	    	  $.ajax({
	 				type: "POST",
	 				async: false,
	 				url: "${pageContext.request.contextPath}/home/analysisRegion.do",
	 				success: 
		   			function(regionlist){
	 					//monthlist.filter(myFunction);
	 					 numbers4 = regionlist.map(myFunction4);
	 					 numbers5 = regionlist.map(myFunction3);
	 					maxdactnocctv=numbers4.indexOf(Math.max(...numbers4));
	 					
	 				}
	      		});
	    	  
	      function myFunction(value, index, array) {
	    	  return value.dactno;
	    	}
	      function myFunction2(value, index, array) {
	    	  return value.dpertime;
	    	}
	      function myFunction3(value, index, array) {
	    	  return value.dfinder;
	    	}
	      function myFunction4(value, index, array) {
	    	  return value.dperfinder;
	    	}
	      function myFunction5(value, index, array) {
	    	  tempnum= array[index].dpertime
	    	  if(tempnum<10){
	    		  tempnum=tempnum.charAt(tempnum.length-1);;
	    	  }
	    	  if(tempnum>=6&&tempnum<=18){
	    		  timedactnoarr[tempnum]={"y":value.dactno,marker: {"symbol": "url(${pageContext.request.contextPath}/resource/img/sun.png)"}}
	    	  }else{
	    		  timedactnoarr[tempnum]={"y":value.dactno,marker: {"symbol": "url(${pageContext.request.contextPath}/resource/img/moon.png)"}}
	    	  }
	    	  return value.dactno;
	    	}
	      function myFunction6(value, index, array) {
	    	  return value.dpertime;
	    	}
	      function chart(){
	      var chart = Highcharts.chart('containerc', {
	    	    title: {
	    	        text: pertranslate[$('#term1').val()]+' 탐지사건 발생수'
	    	    },

	    	    subtitle: {
	    	        text: '가슴이 웅장해진다...'
	    	    },

	    	    xAxis: {
	    	        categories: numbers3
	    	    },

	    	    series: [{
	    	        type: 'column',
	    	        colorByPoint: true,
	    	        data: numbers2,
	    	        showInLegend: false
	    	    }],
	    	    credits:{
	    	    	enabled: false
	    	    },
	    	    exporting:{
	    	    	enabled: false
	    	    }

	    	});}
	      function chart2(){
	      var chart = Highcharts.chart('container2', {
	    	    title: {
	    	        text: '탐지객체별 탐지사건 발생수'
	    	    },

	    	    subtitle: {
	    	        text: termtranslate[$('#term2').val()]+' 기준 가장 위험한 지역은'+ numbers5[maxdactnocctv]+'입니다.'
	    	    },

	    	    xAxis: {
	    	        categories: numbers5
	    	    },

	    	    series: [{
	    	        type: 'column',
	    	        colorByPoint: true,
	    	        data: numbers4,
	    	        showInLegend: false
	    	    }],
	    	    credits:{
	    	    	enabled: false
	    	    },
	    	    exporting:{
	    	    	enabled: false
	    	    }

	    	});
	      }
	      function chart3(){
	      Highcharts.chart('container3', {
	    	    chart: {
	    	        type: 'spline'
	    	    },
	    	    title: {
	    	        text: '시간별 탐지건수'
	    	    },
	    	    subtitle: {
	    	        text: termtranslate[$('#term3').val()]+" 기준"+numbers7[maxdactno]+"시... "+numbers7[maxdactno]+"시를 조심하십시오..."
	    	    },
	    	    xAxis: {
	    	        categories: timearr
	    	    },
	    	    yAxis: {
	    	        title: {
	    	            text: '탐지건수'
	    	        },
	    	        labels: {
	    	            /* formatter: function () {
	    	                return this.value + '°';
	    	            } */
	    	        }
	    	    },
	    	    tooltip: {
	    	        crosshairs: true,
	    	        shared: true
	    	    },
	    	    plotOptions: {
	    	        spline: {
	    	            marker: {
	    	                radius: 4,
	    	                lineColor: '#666666',
	    	                lineWidth: 1
	    	            }
	    	        }
	    	    },
	    	    series: [ {
	    	        name: '탐지',
	    	        marker: {
	    	        	 symbol: "square"
	    	        },
	    	        data: timedactnoarr
	    	    } ],
	    	    credits:{
	    	    	enabled: false
	    	    },
	    	    exporting:{
	    	    	enabled: false
	    	    }
	    	});
	      }
	      console.log(timedactnoarr);
	      console.log(maxdactno);
</script>
		</div>
	</div>

    
    </body>
</html>