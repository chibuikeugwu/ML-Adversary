package com.example.memorymaster.entity;

import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableLogic;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

@Data
public class Pictures {
    private static final long serialVersionUID = 1L;

    @ApiModelProperty("picture id")
    @TableId
    private Long pictureId;

    @ApiModelProperty("picture name")
    private String pictureName;

    @ApiModelProperty("picture label")
    private String pictureLabel;

    @ApiModelProperty("picture like")
    private Integer pictureLike;

    @ApiModelProperty("picture owner id")
    private Long pictureOwnerId;

    @ApiModelProperty("picture file path")
    private String pictureFilePath;

    @TableLogic(delval = "1", value = "0")
    private Integer isDeleted;
}
