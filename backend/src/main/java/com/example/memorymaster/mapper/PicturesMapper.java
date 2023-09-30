package com.example.memorymaster.mapper;

import com.baomidou.mybatisplus.annotation.InterceptorIgnore;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;

import com.example.memorymaster.entity.Pictures;

@InterceptorIgnore(tenantLine = "true")
public interface PicturesMapper extends BaseMapper<Pictures> {
}
