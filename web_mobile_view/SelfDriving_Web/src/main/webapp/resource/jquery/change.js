var bigPic = document.querySelector(".big");
var smallPics = document.querySelectorAll(".small");

for (var i=0; i <smallPics.length ; i++){
	smallPics[i].addEventListener("click", changepic);
}

function changepic(){
	// smallPic의 id 뽑기
	var smallPicAttribute = this.getAttribute("id");
	// bigPic의 id 뽑아서 임시 저장
	var temp = bigPic.getAttribute("id");
	// bigPic의 id를 smallPic의 id로 바꾸기
	bigPic.setAttribute("id", smallPicAttribute);
	// smallPic자리에 bigPic의 id넣기
	this.setAttribute("id", temp);
}