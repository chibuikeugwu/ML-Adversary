package com.example.memorymaster.mapper;

import com.baomidou.mybatisplus.annotation.InterceptorIgnore;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.example.memorymaster.entity.Users;


@InterceptorIgnore(tenantLine = "true")
public interface UsersMapper extends BaseMapper<Users> {

}
