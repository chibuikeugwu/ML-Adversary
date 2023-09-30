package com.example.memorymaster.entity;

import java.io.Serializable;

import com.baomidou.mybatisplus.annotation.TableId;
import io.swagger.annotations.ApiModel;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

/**
 * <p>
 * 
 * </p>
 *
 * @author jiawei
 * @since 2023-04-09
 */
@Data
@ApiModel(value = "Users对象", description = "")
public class Users implements Serializable {

    private static final long serialVersionUID = 1L;

    @ApiModelProperty("user name")
    private String userName;

    @ApiModelProperty("user id")
    @TableId
    private Long userId;

    @ApiModelProperty("user password")
    private String userPassword;

    @ApiModelProperty("user email")
    private String userEmail;

    @ApiModelProperty("user avatar")
    private String userAvatar;


}
