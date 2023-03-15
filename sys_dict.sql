/*
 Navicat Premium Data Transfer

 Source Server         : localhost_Mysql
 Source Server Type    : MySQL
 Source Server Version : 50724
 Source Host           : localhost:3306
 Source Schema         : detect

 Target Server Type    : MySQL
 Target Server Version : 50724
 File Encoding         : 65001

 Date: 09/03/2023 09:41:26
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for sys_dict
-- ----------------------------
DROP TABLE IF EXISTS `sys_dict`;
CREATE TABLE `sys_dict`  (
  `dict_id` int(11) NOT NULL AUTO_INCREMENT,
  `dict_name` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL,
  `dict_name_en` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL,
  `create_time` datetime(6) NOT NULL,
  `update_time` datetime(6) NOT NULL,
  PRIMARY KEY (`dict_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 6 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of sys_dict
-- ----------------------------
INSERT INTO `sys_dict` VALUES (2, 'JX', 'product_name', '2023-02-24 01:08:33.000000', '2023-02-24 01:08:41.378939');
INSERT INTO `sys_dict` VALUES (3, 'PC', 'batch_no', '2023-02-24 01:08:49.000000', '2023-02-24 01:09:18.863939');
INSERT INTO `sys_dict` VALUES (4, 'JC', 'plane_no', '2023-02-24 01:09:20.000000', '2023-02-24 01:09:29.277939');
INSERT INTO `sys_dict` VALUES (5, '对比状态', 'compare_status', '2023-02-24 01:09:34.000000', '2023-02-24 01:09:43.268939');

SET FOREIGN_KEY_CHECKS = 1;
