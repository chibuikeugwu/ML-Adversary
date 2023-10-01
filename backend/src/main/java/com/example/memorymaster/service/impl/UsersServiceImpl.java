package com.example.memorymaster.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.example.memorymaster.entity.Users;
import com.example.memorymaster.entity.params.UserVO;
import com.example.memorymaster.mapper.UsersMapper;
import com.example.memorymaster.service.IUsersService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

/**
 * <p>
 *  服务实现类
 * </p>
 *
 * @author jiawei
 * @since 2023-04-09
 */
@Service
public class UsersServiceImpl extends ServiceImpl<UsersMapper, Users> implements IUsersService {
    @Autowired
    private UsersMapper usersMapper;

    @Override
    public boolean verifyPassword(Users user) {
        QueryWrapper<Users> wrapper=new QueryWrapper<>();
        wrapper.eq("user_email",user.getUserEmail());
        if(usersMapper.selectOne(wrapper)==null){
            return false;//不存在此用户
        }
        //用户账户存在
        String correctPassword = usersMapper.selectOne(wrapper).getUserPassword();
        return correctPassword.equals(user.getUserPassword());
    }


    @Override
    public UserVO getCurrentUserInfo(Users user) {
        QueryWrapper<Users> wrapper=new QueryWrapper<>();
        wrapper.eq("user_email",user.getUserEmail());
        Users userDO=usersMapper.selectOne(wrapper);
        UserVO userVO=new UserVO();
        BeanUtils.copyProperties(userDO,userVO);
        return userVO;
    }

    @Override
    public String getUserNameByUserId(Long userId) {
        return getById(userId).getUserName();
    }
}
