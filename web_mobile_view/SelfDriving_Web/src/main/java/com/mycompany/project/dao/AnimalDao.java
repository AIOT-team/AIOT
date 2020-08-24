package com.mycompany.project.dao;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Repository;

import com.mycompany.project.model.Animal;

import egovframework.rte.psl.dataaccess.EgovAbstractMapper;

@Repository
public class AnimalDao extends EgovAbstractMapper {
	private static final Logger LOGGER = LoggerFactory.getLogger(AnimalDao.class);

	public void insert(Animal animal) {
		//LOGGER.info("db에 insert할거다");
		insert("insert", animal);
	}

	public Animal selectByDno(int dno) {
		//LOGGER.info("db에서 select할거다");
		Animal animal = selectOne("animal.selectByDno",dno);
		return animal;
	}

	public int count() {
		int totalRows = selectOne("animal.count");
		return totalRows;
	}

	public List<Animal> selectByPage(int pageNo, int rowsPerPage) {
		LOGGER.info("실행");
		int startIndex = (pageNo-1) * rowsPerPage;
		Map<String, Integer> map = new HashMap<>();
		map.put("startIndex", startIndex);
		map.put("rowsPerPage", rowsPerPage);
		List<Animal> list = selectList("animal.selectByPage", map);
		return list;
	}

	public List getanalysisMonth() {
		List monthlist = selectList("animal.monthList");
		return monthlist;
	}

	public List getanalysisRegion() {
		List monthlist = selectList("animal.regionList");
		return monthlist;
	}

	public String getmainDangerLevel() {
		String howdanger = selectOne("animal.howdanger");
		return howdanger;
	}

	public List<Animal> selectByCCTV() {
		LOGGER.info("실행");
		List<Animal> list = selectList("animal.selectByCCTV");
		return list; 
	}

	public void updateDcomplete(int dno) {
		int rows = update("animal.updateDcompleteByDno", dno);
	}

	public List getanalysisHour() {
		List hourlist = selectList("animal.hourList");
		return hourlist;
	}

	public List getanalysisHourwithterm(String term) {
		List hourlistwithterm= new ArrayList<>();
		if(term.equals("all")) {
			hourlistwithterm = selectList("animal.hourList");	
		}
		else if(term.equals("oneyear")) {
			hourlistwithterm = selectList("animal.hourListoneyear");
		}else if(term.equals("onemonth")) {
			hourlistwithterm = selectList("animal.hourListonemonth");
		}else if(term.equals("oneweek")) {
			hourlistwithterm = selectList("animal.hourListoneweek");
		}
		return hourlistwithterm;
	}

	public List getanalysisRegionwithterm(String term) {
		List regionlistwithterm= new ArrayList<>();
		if(term.equals("all")) {
			regionlistwithterm = selectList("animal.regionList");	
		}
		else if(term.equals("oneyear")) {
			regionlistwithterm = selectList("animal.regionListoneyear");
		}else if(term.equals("onemonth")) {
			regionlistwithterm = selectList("animal.regionListonemonth");
		}else if(term.equals("oneweek")) {
			regionlistwithterm = selectList("animal.regionListoneweek");
		}
		return regionlistwithterm;
	}

	public List getanalysisMonthwithterm(String termval) {
		List monthlistwithterm= new ArrayList<>();
		if(termval.equals("hour")) {
			monthlistwithterm = selectList("animal.monthList");	
		}
		else if(termval.equals("day")) {
			monthlistwithterm = selectList("animal.monthListday");
		}else if(termval.equals("month")) {
			monthlistwithterm = selectList("animal.monthListmonth");
		}else if(termval.equals("year")) {
			monthlistwithterm = selectList("animal.monthListyear");
		}
		return monthlistwithterm;
	}

	public List getDlevelCount() {
		List dlevelCount = new ArrayList<>();
		dlevelCount = selectList("animal.countLevel");
		return dlevelCount;
	}
}
