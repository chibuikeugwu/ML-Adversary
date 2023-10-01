/*
 Navicat Premium Data Transfer

 Source Server         : localhost_3306
 Source Server Type    : MySQL
 Source Server Version : 80032
 Source Host           : localhost:3306
 Source Schema         : mladversary

 Target Server Type    : MySQL
 Target Server Version : 80032
 File Encoding         : 65001

 Date: 01/10/2023 13:48:42
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for pictures
-- ----------------------------
DROP TABLE IF EXISTS `pictures`;
CREATE TABLE `pictures`  (
  `picture_id` bigint NOT NULL AUTO_INCREMENT,
  `picture_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `picture_label` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT '',
  `picture_like` int NULL DEFAULT 0,
  `picture_file_path` varchar(2048) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `picture_owner_id` bigint NULL DEFAULT NULL,
  `is_deleted` int NULL DEFAULT 0,
  PRIMARY KEY (`picture_id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of pictures
-- ----------------------------
INSERT INTO `pictures` VALUES (1, 'A kitty', 'Cat', 0, 'https://th-thumbnailer.cdn-si-edu.com/bgmkh2ypz03IkiRR50I-UMaqUQc=/1000x750/filters:no_upscale():focal(1061x707:1062x708)/https://tf-cmsv2-smithsonianmag-media.s3.amazonaws.com/filer_public/55/95/55958815-3a8a-4032-ac7a-ff8c8ec8898a/gettyimages-1067956982.jpg', 2001, 0);
INSERT INTO `pictures` VALUES (2, 'puppy', 'label', 0, 'https://hips.hearstapps.com/hmg-prod/images/dog-puppy-on-garden-royalty-free-image-1586966191.jpg?crop=0.752xw:1.00xh;0.175xw,0&resize=1200:*', 2001, 0);
INSERT INTO `pictures` VALUES (3, 'asd', 'label', 0, 'https://www.hindustantimes.com/ht-img/img/2023/08/25/1600x900/international_dog_day_1692974397743_1692974414085.jpg', 2001, 0);
INSERT INTO `pictures` VALUES (4, 'kitten', 'label', 0, 'https://www.google.com.hk/url?sa=i&url=https%3A%2F%2Fwww.theguardian.com%2Flifeandstyle%2F2020%2Fsep%2F05%2Fwhat-cats-mean-by-miaow-japans-pet-guru-knows-just-what-your-feline-friend-wants&psig=AOvVaw34yLSKFoeY-uITpMb8wq0K&ust=1696189619365000&source=images&cd=vfe&opi=89978449&ved=0CBAQjRxqFwoTCLComfGM04EDFQAAAAAdAAAAABAE', 2001, 0);

-- ----------------------------
-- Table structure for users
-- ----------------------------
DROP TABLE IF EXISTS `users`;
CREATE TABLE `users`  (
  `user_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '用户名',
  `user_id` bigint NOT NULL AUTO_INCREMENT COMMENT '用户id',
  `user_password` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '用户密码',
  `user_email` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '用户邮箱',
  `user_avatar` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '用户头像url',
  PRIMARY KEY (`user_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 2004 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of users
-- ----------------------------
INSERT INTO `users` VALUES ('gjw', 2001, '123456', 'guojw7@foxmail.com', NULL);
INSERT INTO `users` VALUES ('Jimmy', 2002, '123456', '819841785@qq.com', NULL);
INSERT INTO `users` VALUES ('Kima', 2003, '123456', 'kima@qq.com', NULL);

SET FOREIGN_KEY_CHECKS = 1;
