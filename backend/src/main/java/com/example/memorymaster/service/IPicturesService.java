package com.example.memorymaster.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.example.memorymaster.entity.Pictures;
import com.example.memorymaster.entity.params.PageParam;

import java.util.List;

public interface IPicturesService extends IService<Pictures> {
    Page<Pictures> getPicturesByUserId(PageParam pageParam);
    int uploadPicture(Pictures picture);
    int updatePictures(List<Pictures> pictures);
}
