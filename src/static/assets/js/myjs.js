function validate_form(thisform){
	
	$("#upload_btn").attr("disabled",true);
	$("#upload_btn").html("处理中")
	$("#upload_btn_vid").attr("disabled",true);
	$("#upload_btn_vid").html("处理中")
	
	return true;
}

function change_range(thisid){
	//alert(thisid.value);
	document.getElementById("label_hpro").innerHTML=thisid.value;
}

function change_range_vid(thisid){
	//alert(thisid.value);
	document.getElementById("label_hpro_vid").innerHTML=thisid.value;
}

var video_ori = document.getElementById("video_ori");
var video_slomo = document.getElementById("video_slomo");

video_ori.ontimeupdate = function () { 
	console.log(this.currentTime);
	console.log(video_slomo.currentTime);
};

video_slomo.ontimeupdate = function () { 
	if (Math.abs(video_ori.currentTime-this.currentTime)>0.1)video_ori.currentTime=this.currentTime;
};

