package com.example.memorymaster.entity.params;

import lombok.Data;

@Data
public class PageParam {
    private Long pageNo=1L;
    private Long pageSize=10L;

    public PageParam(){

    }
    public PageParam(Long _pageNo, Long _pageSize){
        pageNo=_pageNo;
        pageSize=_pageSize;
    }
}
