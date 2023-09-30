package com.example.memorymaster.common.utils;

import com.example.memorymaster.entity.Users;
import com.example.memorymaster.entity.params.UserVO;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import javax.servlet.http.HttpSession;

public class SessionUtils {
    public static HttpSession session() {
        ServletRequestAttributes attr = (ServletRequestAttributes) RequestContextHolder.currentRequestAttributes();
        return attr.getRequest().getSession(true); // true == allow create
    }

    public static void setCurrentUser(UserVO loginUser) {
        session().setAttribute("currentUser", loginUser);
    }
    public static UserVO getCurrentUser() {
        return (UserVO) session().getAttribute("currentUser");
    }
}