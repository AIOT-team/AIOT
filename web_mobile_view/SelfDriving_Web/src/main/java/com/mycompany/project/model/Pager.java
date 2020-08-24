package com.mycompany.project.model;

public class Pager {

		private int totalRows;						// 전체 행수
		private int totalPageNo;					// 전체 페이지 수
		private int totalGroupNo;					// 전체 그룹 수
		private int startPageNo;					// 그룹의 시작 페이지 번호
		private int endPageNo;						// 그룹의 끝 페이지 번호
		private int pageNo;							// 현재 페이지 번호
		private int groupNo;						// 현재 그룹 번호
		private int rowsPerPage;					// 페이지 당 행수
		private int pagesPerGroup;					// 그룹 당 페이지 수
		
		public Pager(int rowsPerPage, int pagesPerGroup, int totalRows, int pageNo) {
			this.rowsPerPage = rowsPerPage;
			this.pagesPerGroup = pagesPerGroup;
			this.totalRows = totalRows;
			this.pageNo = pageNo;
			
			//전체 페이지 수
			totalPageNo = totalRows / rowsPerPage;
			if(totalRows % rowsPerPage != 0) totalPageNo++;
			
			//전체 그룹 수
			totalGroupNo = totalPageNo / pagesPerGroup;
			if(totalPageNo%pagesPerGroup != 0) totalGroupNo++;
			
			//현재 그룹 번호
			groupNo = (pageNo-1) / pagesPerGroup + 1;
			
			//그룹의 시작 페이지 번호
			startPageNo = (groupNo-1) * pagesPerGroup + 1;
			
			//그룹의 끝 페이지 번호
			endPageNo = startPageNo + pagesPerGroup - 1;
			if(groupNo == totalGroupNo) endPageNo = totalPageNo;
		}

		public int getTotalRows() {
			return totalRows;
		}

		public int getTotalPageNo() {
			return totalPageNo;
		}

		public int getTotalGroupNo() {
			return totalGroupNo;
		}

		public int getStartPageNo() {
			return startPageNo;
		}

		public int getEndPageNo() {
			return endPageNo;
		}

		public int getPageNo() {
			return pageNo;
		}

		public int getGroupNo() {
			return groupNo;
		}

		public int getRowsPerPage() {
			return rowsPerPage;
		}

		public int getPagesPerGroup() {
			return pagesPerGroup;
		}
}