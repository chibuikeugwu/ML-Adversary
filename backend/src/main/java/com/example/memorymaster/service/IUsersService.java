package com.example.memorymaster.service;

import com.example.memorymaster.entity.Users;
import com.baomidou.mybatisplus.extension.service.IService;
import com.example.memorymaster.entity.params.UserVO;

import java.io.Serializable;


public interface IUsersService extends IService<Users> {
    boolean verifyPassword(Users user);
    UserVO getCurrentUserInfo(Users users);
    String getUserNameByUserId(Long userId);
}
