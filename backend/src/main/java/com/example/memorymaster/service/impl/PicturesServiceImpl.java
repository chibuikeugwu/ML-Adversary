package com.example.memorymaster.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.example.memorymaster.common.utils.SessionUtils;
import com.example.memorymaster.entity.Pictures;
import com.example.memorymaster.entity.params.PageParam;
import com.example.memorymaster.mapper.PicturesMapper;
import com.example.memorymaster.service.IPicturesService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class PicturesServiceImpl extends ServiceImpl<PicturesMapper, Pictures> implements IPicturesService {
    @Autowired
    private PicturesMapper picturesMapper;

    @Override
    public Page<Pictures> getPicturesByUserId(PageParam pageParam) {
        Page<Pictures> page=new Page<>(pageParam.getPageNo(), pageParam.getPageSize());
        QueryWrapper<Pictures> wrapper=new QueryWrapper<>();
        wrapper.eq("picture_owner_id", SessionUtils.getCurrentUser().getUserId()).eq("is_deleted",0);
        return picturesMapper.selectPage(page,wrapper);
    }

    @Override
    public int uploadPicture(Pictures picture) {
        QueryWrapper<Pictures> wrapper=new QueryWrapper<>();
        return picturesMapper.insert(picture);
    }
}
