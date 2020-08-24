package com.mycompany.project.model;

import java.util.Date;

public class Animal {

	private int dno;
	private int dactno;
	private String dname;	
	private String dlevel;
	private int dnum;	
	private String dfinder;	
	private String dlocation;
	private Date dtime;
	private String dimagesname;			
	private String dimagetype;
	private int dcomplete;
	private String dzone;
	
	
	public int getDactno() {
		return dactno;
	}
	public void setDactno(int dactno) {
		this.dactno = dactno;
	}
	public String getDzone() {
		return dzone;
	}
	public void setDzone(String dzone) {
		this.dzone = dzone;
	}
	public int getDno() {
		return dno;
	}
	public void setDno(int dno) {
		this.dno = dno;
	}
	public String getDname() {
		return dname;
	}
	public void setDname(String dname) {
		this.dname = dname;
	}
	public String getDlevel() {
		return dlevel;
	}
	public void setDlevel(String dlevel) {
		this.dlevel = dlevel;
	}
	public int getDnum() {
		return dnum;
	}
	public void setDnum(int dnum) {
		this.dnum = dnum;
	}
	public String getDfinder() {
		return dfinder;
	}
	public void setDfinder(String dfinder) {
		this.dfinder = dfinder;
	}
	public String getDlocation() {
		return dlocation;
	}
	public void setDlocation(String dlocation) {
		this.dlocation = dlocation;
	}
	public Date getDtime() {
		return dtime;
	}
	public void setDtime(Date dtime) {
		this.dtime = dtime;
	}
	public String getDimagesname() {
		return dimagesname;
	}
	public void setDimagesname(String dimagesname) {
		this.dimagesname = dimagesname;
	}
	public String getDimagetype() {
		return dimagetype;
	}
	public void setDimagetype(String dimagetype) {
		this.dimagetype = dimagetype;
	}
	public int getDcomplete() {
		return dcomplete;
	}
	public void setDcomplete(int dcomplete) {
		this.dcomplete = dcomplete;
	}
}