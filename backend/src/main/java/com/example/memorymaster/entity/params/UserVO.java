package com.example.memorymaster.entity.params;

import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

@Data
public class UserVO {
    private static final long serialVersionUID = 1L;

    @ApiModelProperty("user name")
    private String userName;

    @ApiModelProperty("user id")
    private Long userId;

    @ApiModelProperty("user mail")
    private String userEmail;

    @ApiModelProperty("user avatar")
    private String userAvatar;
}
