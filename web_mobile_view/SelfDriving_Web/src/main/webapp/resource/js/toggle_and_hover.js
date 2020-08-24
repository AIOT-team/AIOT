/*<style type="text/css">
    .toggleBG{background: #CCCCCC; width: 70px; height: 30px; border: 1px solid #CCCCCC; border-radius: 15px;}
    .toggleFG{background: #FFFFFF; width: 30px; height: 30px; border: none; border-radius: 15px; position: relative; left: 0px;}
</style>
// togle css 부분
*/

$(document).on('click', '.toggleBG', function () {
	    var toggleBG = $(this);
	    var toggleFG = $(this).find('.toggleFG');
	    var left = toggleFG.css('left');
	    if(left == '40px') {
	        toggleBG.css('background', '#CCCCCC');
	        toggleActionStart(toggleFG, 'TO_LEFT');
	    }else if(left == '0px') {
	        toggleBG.css('background', '#53FF4C');
	        toggleActionStart(toggleFG, 'TO_RIGHT');
	    }
	});
	 
	// 토글 버튼 이동 모션 함수
	function toggleActionStart(toggleBtn, LR) {
	    // 0.01초 단위로 실행
	    var intervalID = setInterval(
	        function() {
	            // 버튼 이동
	            var left = parseInt(toggleBtn.css('left'));
	            left += (LR == 'TO_RIGHT') ? 5 : -5;
	            if(left >= 0 && left <= 40) {
	                left += 'px';
	                toggleBtn.css('left', left);
	            }
	        }, 10);
	    setTimeout(function(){
	        clearInterval(intervalID);
	    }, 201);
	};
	
	function getToggleBtnState(toggleBtnId){
	    const left_px = parseInt( $('#'+toggleBtnId).css('left') );
	    //return (left_px > 0)? "on" : "off";
	    if(left_px > 0){
	    	$(".button_menual1").hide();
	    	$(".button_menual2").show();
	    }else{
	    	//text_data = $(".button_menual").html()
	    	$(".button_menual2").hide();
	    	$(".button_menual1").show();
	    };		
	    	
	}
	
/*// 이걸 갔다 붙여넣음.
<div id = "button_state_mode">
	<div class="button_menual1" style="display: none;">		
				<p>--------------------수동 주행 메뉴얼--------------------<p><br>
				<p>1. 키보드 상하좌우 = 차의 이동</p>	
	</div>
	<div class="button_menual2">
				<p>--------------------자율 주행 메뉴얼--------------------</p><br>
				<p>1. 거리가 10 이하가 되면 운행을 중지하고 뒤로 갑니다<br>
				<p>2. 이후 추가 바람.</p><br>
	</div>
</div>
*/
	
	
//////////////////////////////////////////////////////////////////////////
	
/*    <style>
    .chart_container {
    	overflow:hidden;
        -webkit-transform:scale(1);
        -moz-transform:scale(1);
        -ms-transform:scale(1); 
        -o-transform:scale(1);  
        transform:scale(1);
        -webkit-transition:.3s;
        -moz-transition:.3s;
        -ms-transition:.3s;
        -o-transition:.3s;
        transition:.3s;
    }
    .chart_container:hover{
    	margin-top:100px;
        -webkit-transform:scale(1.2);
        -moz-transform:scale(1.2);
        -ms-transform:scale(1.2);   
        -o-transform:scale(1.2);
        transform:scale(1.2);
    }
             </style>
             
             * 차트가 그려지는 곳에 chart_container를 클래스로 줄 것.
             */