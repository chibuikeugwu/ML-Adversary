package com.example.memorymaster.controller;

import com.example.memorymaster.common.JsonResponse;
import com.example.memorymaster.common.utils.SessionUtils;
import com.example.memorymaster.entity.Pictures;
import com.example.memorymaster.entity.params.PageParam;
import com.example.memorymaster.service.IPicturesService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;

@Controller
@RequestMapping("/pictures")
public class PictureController {
    @Autowired
    private IPicturesService picturesService;

    @ResponseBody
    @GetMapping("/")
    public JsonResponse getMyPictures(PageParam pageParam){
        return JsonResponse.success(picturesService.getPicturesByUserId(pageParam));
    }

    @ResponseBody
    @GetMapping("/upload")
    public JsonResponse uploadPicture(Pictures picture){
        return JsonResponse.success(picturesService.uploadPicture(picture));
    }
}
