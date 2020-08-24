package com.mycompany.project.controller;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.UnknownHostException;

import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.annotation.PostConstruct;

import org.apache.tomcat.jni.Time;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpRequest;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.util.FileCopyUtils;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;

import com.mycompany.project.service.MQTT;
import com.mycompany.project.model.Animal;
import com.mycompany.project.model.Pager;
import com.mycompany.project.service.AnimalService;
import javax.servlet.http.HttpServletRequest;

@Controller
@RequestMapping("/home") 
public class HomeController {
	private static final Logger LOGGER = LoggerFactory.getLogger(HomeController.class);
	
	@Autowired
	private AnimalService animalService;
	
	@Autowired
	MQTT ReadFromOtherMQTT;
	
	//MQTT 연결 및 스레드 실행
	@PostConstruct
	public void mqttConnect() {
		String MqttServer1= "tcp://192.168.3.105:1883";
		String client_id = "hostname";
		String username = "hostname";	
		String passwd = "12345";
		String topic = "/*";
	
		ReadFromOtherMQTT.chogihwa(MqttServer1, client_id, username, passwd);
		ReadFromOtherMQTT.init(topic);
		ReadFromOtherMQTT.subscribe(0);

		//MQTT mqtt = new MQTT();
		//mqtt.start();
		ReadFromOtherMQTT.setDaemon(true);
		ReadFromOtherMQTT.start();
	}
	
	@RequestMapping("/landing.do")
	public String landing(){
		LOGGER.info("실행");
		return "home/landing";
	}
	
	@RequestMapping("/main.do")
	public String main(){
		return "home/main";
	}
	
	@RequestMapping("/hud.do")
	public String hud(){
		return "home/hud";
	}
	
	@RequestMapping("/jetracer.do")
	public String jetbot(){
		LOGGER.info("실행");
		return "home/jetracer";
	}
	//history.jsp에서 이미지를 보여준다.
	@RequestMapping("/imageView.do")
	@ResponseBody
	public void imageView(@RequestParam int dno,
						HttpServletResponse response) throws Exception {

		Animal animal = new Animal();
		animal = animalService.getAnimal(dno);
		//LOGGER.info("이미지 뽑았다ㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏ");
		
		String imgLoc = animal.getDlocation();
		LOGGER.info(imgLoc);
		
		InputStream is = new FileInputStream(imgLoc);
		OutputStream os = response.getOutputStream();
		FileCopyUtils.copy(is, os);
		os.close();
		is.close();
	}
	//status.jsp에서 리스트를 갖고온다. 	
	@RequestMapping("/status.do")
	public String status(Model model){
		LOGGER.info("실행");
		List<Animal>list = animalService.getCCTVanimal();
		model.addAttribute("animal", list);
		return "home/status";
	}
	
	//
	@ResponseBody
	@RequestMapping("/dcompleteUpdate.do")
	public void update(@RequestParam Map<String,Object> data){
		LOGGER.info("실행");
		Object dnoStr = data.get("dno");
		int dno = Integer.parseInt((String) dnoStr);
		System.out.println("넘어온 사건 번호: " + dno);
		animalService.updateDcomplete(dno);
		System.out.println("업데이트 처리 완료했지롱");
	}
	
	@RequestMapping("/analysis.do")
	public String analysis(){
		LOGGER.info("실행");
		return "home/analysis";
	}
	//페이지처리
	@RequestMapping("/intro.do")
	public String intro(){
		LOGGER.info("실행");
		return "home/intro";
	}
	
	@RequestMapping("/1jet.do")
	public String jet1(){
		return "home/1jet";
	}
	
	@RequestMapping("/2jet.do")
	public String jet2(){
		return "home/2jet";
	}
	
	@RequestMapping("/3jet.do")
	public String jet3(){
		return "home/3jet";
	}

	//페이지처리
	@RequestMapping("/history.do")
	public String history(Model model, @RequestParam(defaultValue="1")int pageNo, 
						@RequestParam(defaultValue="7") int rowsPerPage,
						HttpSession httpSession) {
		LOGGER.info("실행");

		Pager pager = new Pager(rowsPerPage, 10, animalService.getTotalListNo(), pageNo);
		model.addAttribute("pager", pager);
		httpSession.setAttribute("pager", pager);

		model.addAttribute("animal", animalService.getListByPage(pageNo,rowsPerPage));
		return "home/history";
	}
	
	@RequestMapping("/newfile.do")
	public String newfile() {
		LOGGER.info("실행");
		return "home/NewFile";
	}

	@RequestMapping("/sleep.do")
	@ResponseBody
	public void sleep() {
		try {
			//주어진 time millisecond만큼 잠든다.
			Thread.sleep(1000);
		} catch (InterruptedException e) {
			// 트라이-캐치
			e.printStackTrace();
		}
	}	

	@RequestMapping("/analysisMonth.do")
	@ResponseBody
	public List analysisMonth() {
		List monthlist = new ArrayList<>(); 
		monthlist = animalService.getanalysisMonth();
		return monthlist;
	}
	
	@RequestMapping("/analysisMonthwithterm.do")
	@ResponseBody
	public List analysisMonth(@RequestBody String termval) {
		termval= termval.replaceAll("=", "");
		List monthlistwithterm = new ArrayList<>(); 
		monthlistwithterm = animalService.getanalysisMonthwithterm(termval);
		return monthlistwithterm;
	}
	
	@RequestMapping("/analysisHour.do")
	@ResponseBody
	public List analysisHour() {
		List hourlist = new ArrayList<>(); 
		hourlist = animalService.getanalysisHour();
		return hourlist;
	}
	
	@RequestMapping("/analysisHourwithterm.do")
	@ResponseBody
	public List analysisHourwithterm(@RequestBody String termval){
		LOGGER.info(termval);
		termval= termval.replaceAll("=", "");
		List hourlistwithterm = new ArrayList<>(); 
		hourlistwithterm = animalService.getanalysisHourwithterm(termval);
		return hourlistwithterm;
	}
	
	@RequestMapping("/analysisRegion.do")
	@ResponseBody
	public List analysisRegion() {
		List regionlist = new ArrayList<>(); 
		regionlist = animalService.getanalysisRegion();
		return regionlist;
	}
	
	@RequestMapping("/analysisRegionwithterm.do")
	@ResponseBody
	public List analysisRegionwithterm(@RequestBody String termval){
		LOGGER.info(termval);
		termval= termval.replaceAll("=", "");
		List regionlistwithterm = new ArrayList<>(); 
		regionlistwithterm = animalService.getanalysisRegionwithterm(termval);
		return regionlistwithterm;
	}
	
	@RequestMapping("/mainDangerLevel.do")
	@ResponseBody
	public String mainDangerLevel() {
		String howdanger = animalService.mainDangerLevel();
		return howdanger;
	}
	
	@RequestMapping("/LevelCount.do")
	@ResponseBody
	public List mainLevelCount() {
		List dlevelCount = new ArrayList<>();
		dlevelCount = animalService.levelCount();
		return dlevelCount;
	}
}