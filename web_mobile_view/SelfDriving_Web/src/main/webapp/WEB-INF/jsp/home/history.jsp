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
	    <script src="${pageContext.request.contextPath}/resource/jquery-ui/jquery-ui.min.js"></script>
		<script src="https://cdnjs.cloudflare.com/ajax/libs/paho-mqtt/1.0.1/mqttws31.min.js" type="text/javascript"></script>
		
	    <link rel="stylesheet" href="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/vendor/bootstrap/css/bootstrap.min.css">
	    <link rel="stylesheet" href="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/css/font.css">
	    <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Muli:300,400,700">
	    <link rel="stylesheet" href="https://d19m59y37dris4.cloudfront.net/dark-admin/1-4-6/css/style.default.css" id="theme-stylesheet">
		<link rel=icon href="${pageContext.request.contextPath}/resource/img/jetracer.png">
		<link rel="stylesheet" href="${pageContext.request.contextPath}/resource/css/yunjis.css">
		
		<script src="${pageContext.request.contextPath}/resource/jquery/jquery-3.4.1.min.js" crossorigin="anonymous"></script>
		<script src="${pageContext.request.contextPath}/resource/popper/popper.min.js"></script>
		
		<link rel="stylesheet" href="${pageContext.request.contextPath}/resource/jquery-ui/jquery-ui.min.css">
		<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">	
		<script src="${pageContext.request.contextPath}/resource/js/moment.min.js"></script>

		<script>
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
					location.href="jetracer.do";
				}else if (obj.direction=="right"){
					location.href="status.do";
				}
			}
}
			$(document).ready(function() {
			    setInterval(renew, 10000);
			});
			
			function renew(){
				location.reload();
			}
			  /* $.ajax({
			   type: "POST",
			   url: "${pageContext.request.contextPath}/home/getAnimalList.do",
			   success: 
				   function(animallist){
				   $('#dataframe').empty();
				   
		   			animallist.forEach(function (item, index, array) {
		    	    var rowItem = "<tr>"
		    	    rowItem += "<td>"+item['dno']+"</td>"
		    	    rowItem += "<td onclick="+ "\"" + "viewImage(" + item['dno'] + ")" + "\"" + "style=" + "\"" + "color: #DB6574; font-weight: 500" + "\">" + item['dname']+"</td>"
		    	    rowItem += "<td>"+item['dnum']+"</td>"
		    	    rowItem += "<td>"+item['dfinder']+"</td>"
		    	    rowItem += "<td>"+item['dfinder']+"</td>"
		    	    rowItem += "<td>"+moment(item['dtime']).format("YYYY-MM-DD HH:mm:ss")+"</td>"		    	    
					rowItem += "</tr>"
					
		    	    $('#append_table').append(rowItem);
			    });
			   }
			 });
			} */

			function viewImage(imgDno){
				console.log("출력하고 싶은 이미지 번호:",imgDno);
				$("#imgShow").attr("src", "${pageContext.request.contextPath}/home/imageView.do?dno="+ imgDno)
 			}
			
			function viewImage2(imgDno){
				console.log("출력하고 싶은 이미지 번호:",imgDno);
				$("#imgShow2").attr("src", "${pageContext.request.contextPath}/home/imageView.do?dno="+ imgDno)
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

		<div class="d-flex align-items-stretch"  style="height: 875px;">
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
	          <li><a href="${pageContext.request.contextPath}/home/main.do" style="color: lightgray"> <i class="icon-home"></i>메인 페이지 </a></li>
	          <li><a href="${pageContext.request.contextPath}/home/jetracer.do" style="color: lightgray"> <i class="icon-writing-whiteboard"></i>탐지봇 현황 </a></li>
	          <li class="active"><a href="${pageContext.request.contextPath}/home/history.do" style="color: lightgray"> <i class="icon-grid"></i>탐지 히스토리 조회 </a></li>
	          <li><a href="${pageContext.request.contextPath}/home/status.do" style="color: lightgray"> <i class="icon-padnote"></i>실시간 탐지 | 대응 현황</a></li>
	          <li><a href="${pageContext.request.contextPath}/home/analysis.do" style="color: lightgray"> <i class="icon-chart"></i>탐지 결과 분석 </a></li>
	      	</ul>
	      </nav>
	      
	      <div class="page-content" style="top: -50px; height: 1080px; padding-bottom: 0px; ">
	        <section class="no-padding-top" style="padding: 0px">
	          <div class="container-fluid">

	            <div class="tab-content" id="pills-tabContent" style="height: 974px; margin-left: 25px; margin-right: 25px;  margin-top: 80px ;color: dimgray; border-style:solid; border-width:medium;">
	            <div class="row">
	             <input style="background-color: #ADFF2F; font-size: xx-large;font-weight: bolder; text-align: center; color: black; margin-left: 45px; margin-top:30px;border-color:#ADFF2F ; border-style:solid; border-width:medium;width: 1485px; height: 45px" readonly="readonly" value="유해 동물 탐지 히스토리 조회 및 탐지 이미지 확인">
	              <div class="col-lg-6">
	                <div class="block" style="width: 870px; margin-top: 0px; margin-bottom: 0px; height: 400px; margin-left: 30px" > 
	                  <div class="title" style="color: white"><strong>유해 동물 탐지 | 히스토리</strong></div>
	                  
	                  <div class="table-responsive" style="height: 350px"> 
	                    <table class="table table-striped table-sm" id= "append_table" style="color: white; text-align: center">
	                      <thead style="border-style:double ; border-left: hidden; border-right: hidden; border-top: hidden; border-color: white">
	                        <tr>
	                          <th>사건 번호</th>
	                          <th>동물 명</th>
	                          <th>유해등급</th>
	                          <th>동물 수</th>
	                          <th>탐지 주체</th>
	                          <th>탐지 구역</th>
	                          <th>탐지 시각</th>
	                        </tr>
	                      </thead>
	                      <tbody>
		                      <c:forEach var="animal" items="${animal}">
		                      	<tr>
		                          <td scope="row">${animal.dno}</td>
		                          <td onclick="viewImage(${animal.dno})" style="color: #ADFF2F; font-weight: bold">${animal.dname}</td>
		                          <td>${animal.dlevel}</td>
		                          <td>${animal.dnum}</td>
		                          <td>${animal.dfinder}</td>
		                          <td>${animal.dzone}</td>
		                          <td><fmt:formatDate value="${animal.dtime}" pattern="YYYY-MM-dd HH:mm:ss"/></td>
		                        </tr>
	                      	  </c:forEach>
	                      </tbody>
	                      	<tr>
								<td colspan= "7" style="text-align:center;">
									<a class="btn btn-secondary btn-sm" href="${pageContext.request.contextPath}/home/history.do?pageNo=1"> 처음 </a>
									<c:if test="${pager.groupNo > 1 }">
										<a class="btn btn-outline-secondary btn-sm" href="${pageContext.request.contextPath}/home/history.do?pageNo=${pager.startPageNo - pager.pagesPerGroup}"> 이전 </a>
									</c:if>
									<c:forEach var="i" begin="${pager.startPageNo}" end="${pager.endPageNo}">
										<c:if test="${pager.pageNo == i}">
											<a class="btn btn-danger btn-sm" href="${pageContext.request.contextPath}/home/history.do?pageNo=${i}" style="background-color:#ADFF2F; border-color: #ADFF2F; color: black ">${i}</a>
										</c:if>

										<c:if test="${pager.pageNo != i}">
											<a class="btn btn-outline-secondary btn-sm" href="${pageContext.request.contextPath}/home/history.do?pageNo=${i}">${i}</a>
										</c:if>
									</c:forEach>
									<c:if test="${pager.groupNo < pager.totalGroupNo }">
										<a class="btn btn-secondary btn-sm" href="${pageContext.request.contextPath}/home/history.do?pageNo=${pager.endPageNo + 1 }">다음</a> 
									</c:if>
									<a class="btn btn-secondary btn-sm" href="${pageContext.request.contextPath}/home/history.do?pageNo=${pager.totalPageNo}"> 맨끝 </a>
								</td>
							</tr>
	                    </table>
	                  </div>
	                </div>
	              </div>
	             
	              <div class="col-lg-6">
	                <div class="block" style="width: 600px; height:400px ;margin-top: 0px; margin-bottom: 10px; margin-left: 130px;">
	                  <div class="title" style="color: white"><strong>유해 동물 탐지 | 이미지</strong></div>
	                  <div class="table-responsive" style="justify-content: center;">
	                    <img id="imgShow" src="${pageContext.request.contextPath}/resource/img/farm.png" style="max-width:100%; width:580px ; height: 310px"/>
	                  </div>
	                </div>
	              </div>

 				<input style="background-color: #864DD9; font-size: xx-large; font-weight: bolder; text-align: center; color: white; margin-left: 45px; border-color: transparent; width: 1485px; height: 50px" readonly="readonly" value="주행 사인 탐지 히스토리 조회 및 탐지 이미지 확인">
	              <div class="col-lg-6">
	                <div class="block" style="width: 870px; margin-top: 0px; margin-bottom: 10px; height: 400px; margin-left: 30px">
	                  <div class="title" style="color: white"><strong>주행 사인 탐지 | 히스토리</strong></div>
	                  <div class="table-responsive" style="height: 350px"> 
	                    <table class="table table-striped table-sm" id= "append_table2" style="color: white; text-align: center">
	                      <thead style="border-style:double ; border-left: hidden; border-right: hidden; border-top: hidden; border-color: white">
	                        <tr>
	                          <th>사건 번호</th>
	                          <th>주행 사인 종류</th>
	                          <th>탐지 수</th>
	                          <th>탐지 주체</th>
	                          <th>탐지 구역</th>
	                          <th>탐지 시각</th>
	                        </tr>
	                      </thead>
	                      
	                      <tbody id="dataframe2">
		                      <c:forEach var="animal" items="${animal}">
		                      	<tr>
		                          <td scope="row">${animal.dno}</td>
		                          <td onclick="viewImage2(${animal.dno})" style="color: #864DD9; font-weight: bold;">${animal.dname}</td>
		                          <td>${animal.dnum}</td>
		                          <td>${animal.dfinder}</td>
		                          <td>${animal.dzone}</td>
		                          <td><fmt:formatDate value="${animal.dtime}" pattern="YYYY-MM-dd HH:mm:ss"/></td>
		                        </tr>
	                      	  </c:forEach>
	                      </tbody>
	                      <tr>
							<td colspan= "7" style="text-align:center;">
								<a class="btn btn-secondary btn-sm" href="${pageContext.request.contextPath}/home/history.do?pageNo=1"> 처음 </a>
								<c:if test="${pager.groupNo > 1 }">
									<a class="btn btn-outline-secondary btn-sm" href="${pageContext.request.contextPath}/home/history.do?pageNo=${pager.startPageNo - pager.pagesPerGroup}"> 이전 </a>
								</c:if>
								<c:forEach var="i" begin="${pager.startPageNo}" end="${pager.endPageNo}">
									<c:if test="${pager.pageNo == i}">
										<a class="btn btn-danger btn-sm" href="${pageContext.request.contextPath}/home/history.do?pageNo=${i}" style="background-color:#864DD9; border-color: #864DD9; color: white ">${i}</a>
									</c:if>

									<c:if test="${pager.pageNo != i}">
										<a class="btn btn-outline-secondary btn-sm" href="${pageContext.request.contextPath}/home/history.do?pageNo=${i}">${i}</a>
									</c:if>
								</c:forEach>
								<c:if test="${pager.groupNo < pager.totalGroupNo }">
									<a class="btn btn-secondary btn-sm" href="${pageContext.request.contextPath}/home/history.do?pageNo=${pager.endPageNo + 1 }">다음</a> 
								</c:if>
								<a class="btn btn-secondary btn-sm" href="${pageContext.request.contextPath}/home/history.do?pageNo=${pager.totalPageNo}"> 맨끝 </a>
							</td>
						</tr>
	                   </table>
	                  </div>
	                </div>
	              </div>
	              
	              <div class="col-lg-6">
	                <div class="block" style="width: 600px; height:400px ;margin-top: 0px; margin-bottom: 10px; margin-left: 130px">
	                  <div class="title" style="color: white"><strong>주행 사인 탐지  | 이미지</strong></div>
	                  <div class="table-responsive">
	                    <img id="imgShow2" src="${pageContext.request.contextPath}/resource/img/driving.png" style="height: 310px; max-width:100%; width: 580px; align-self: center;"/>
	                  </div>
	                </div>
	              </div>
	              
	              </div>
	             </div>
	            </div>
			</section>
		</div>
	</div>
</body>
</html>