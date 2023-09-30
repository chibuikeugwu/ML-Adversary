package com.example.memorymaster;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
@MapperScan("com.example.memorymaster.mapper")
public class MemorymasterApplication {

    public static void main(String[] args) {
        SpringApplication.run(MemorymasterApplication.class, args);
    }

}
