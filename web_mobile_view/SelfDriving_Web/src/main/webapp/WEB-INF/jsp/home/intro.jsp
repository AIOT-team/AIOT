
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>AIOT Project | TEAM 2</title>
    <link rel="stylesheet" type="text/css" href="${pageContext.request.contextPath}/resource/assets/css/bootstrap.min.css" >
    <link rel="stylesheet" type="text/css" href="${pageContext.request.contextPath}/resource/assets/fonts/line-icons.css">
    <link rel="stylesheet" type="text/css" href="${pageContext.request.contextPath}/resource/assets/css/slicknav.css">
    <link rel="stylesheet" type="text/css" href="${pageContext.request.contextPath}/resource/assets/css/menu_sideslide.css">
    <link rel="stylesheet" type="text/css" href="${pageContext.request.contextPath}/resource/assets/css/vegas.min.css">
    <link rel="stylesheet" type="text/css" href="${pageContext.request.contextPath}/resource/assets/css/animate.css">
    <link rel="stylesheet" type="text/css" href="${pageContext.request.contextPath}/resource/assets/css/main.css">
    <link rel="stylesheet" type="text/css" href="${pageContext.request.contextPath}/resource/assets/css/responsive.css">
    <link rel=icon href="${pageContext.request.contextPath}/resource/img/jetracer.png">
  </head>
  <body>
    <div class="bg-wraper overlay has-vignette">
      <div id="example" class="slider opacity-50 vegas-container" style="height: 1080px"></div>
    </div>

    <section class="countdown-timer">
      <div class="container">
        <div class="row text-center">
          <div class="col-md-12 col-sm-12 col-xs-12">
            <div class="heading-count">
              <h2 style="margin-top: 100px; background-color: dimgray; opacity: 0.8; height: 100px; justify-content: center; text-align: center; padding-top: 30px; font-size: 50px">HI HI HI BYE BYE BYE</h2>
            </div>
          </div>
          <div class="col-md-12 col-sm-12 col-xs-12">
            <p style="font-size:large">
           	 HI HI HI will allow you to join smart-farming life by protecting your farm 24hours from wild animals<br>
             Please Join Us!
            </p>
            <div class="button-group">
              <a href="${pageContext.request.contextPath}/home/main.do" class="btn btn-common" style="margin-top: 50px; background-color: #ADFF2F; color: black; font-weight: bold; ">Admin Page</a>
              <a href="${pageContext.request.contextPath}/home/main.do" class="btn btn-border" style="margin-top: 50px; background-color: #864DD9; color: white; font-weight: bold; border-color: #864DD9">About us</a>
            </div>
          </div>
        </div>
      </div>
    </section>

<div></div>
    <div id="preloader">
      <div class="loader" id="loader-1"></div>
    </div>

    <script src="${pageContext.request.contextPath}/resource/assets/js/jquery-min.js"></script>
    <script src="${pageContext.request.contextPath}/resource/assets/js/popper.min.js"></script>
    <script src="${pageContext.request.contextPath}/resource/assets/js/bootstrap.min.js"></script>
    <script src="${pageContext.request.contextPath}/resource/assets/js/vegas.min.js"></script>
    <script src="${pageContext.request.contextPath}/resource/assets/js/jquery.countdown.min.js"></script>
    <script src="${pageContext.request.contextPath}/resource/assets/js/classie.js"></script>
    <script src="${pageContext.request.contextPath}/resource/assets/js/jquery.nav.js"></script>
    <script src="${pageContext.request.contextPath}/resource/assets/js/jquery.easing.min.js"></script> 
    <script src="${pageContext.request.contextPath}/resource/assets/js/wow.js"></script>
    <script src="${pageContext.request.contextPath}/resource/assets/js/jquery.slicknav.js"></script>
    <script src="${pageContext.request.contextPath}/resource/assets/js/main.js"></script>
    <script src="${pageContext.request.contextPath}/resource/assets/js/form-validator.min.js"></script>
    <script src="${pageContext.request.contextPath}/resource/assets/js/contact-form-script.min.js"></script>

    <script>
      $("#example").vegas({
          timer: false,
          delay: 6000,
          transitionDuration: 2000,
          transition: "blur",
          slides: [
              { src: "${pageContext.request.contextPath}/resource/img/farm.jpg" },
              { src: "${pageContext.request.contextPath}/resource/img/land.jpg" },
              { src: "${pageContext.request.contextPath}/resource/img/farm2.jpg" },
          ]
      });
    </script>
      
  </body>
</html>