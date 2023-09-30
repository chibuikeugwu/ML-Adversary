package com.example.memorymaster.controller;

import com.example.memorymaster.common.JsonResponse;
import com.example.memorymaster.common.utils.SessionUtils;
import com.example.memorymaster.entity.Users;
import com.example.memorymaster.entity.params.UserVO;
import com.example.memorymaster.service.IUsersService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.stereotype.Controller;


@Controller
@RequestMapping("/users")
public class UsersController {

    @Autowired
    private IUsersService usersService;

    @ResponseBody
    @PostMapping("/login")
    public JsonResponse login(@RequestBody Users user){
        if(usersService.verifyPassword(user)){
            UserVO currentUser=usersService.getCurrentUserInfo(user);
            SessionUtils.setCurrentUser(currentUser);
            return JsonResponse.success(currentUser,"Login successfully");
        }
        return JsonResponse.failure("Wrong email or password");
    }

    @ResponseBody
    @GetMapping("/username")
    public JsonResponse getUserName(Long userId){
        return JsonResponse.success(usersService.getUserNameByUserId(userId));
    }
}
