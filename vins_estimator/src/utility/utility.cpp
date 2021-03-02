/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "utility.h"
#include <iostream>

/**
 * 计算g对齐到重力加速度方向所需的旋转
*/
Eigen::Matrix3d Utility::g2R(const Eigen::Vector3d &g)
{
    Eigen::Matrix3d R0;
    Eigen::Vector3d ng1 = g.normalized();
    // ENU世界坐标系下的z方向（重力加速度方向）
    Eigen::Vector3d ng2{0, 0, 1.0};
    // ng1为当前IMU坐标系的z方向，R0 * ng1 = ng2，表示当前IMU坐标系在世界系中的姿态，或者说R0可以将当前IMU坐标系点变换成世界坐标系点
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();

    // 测试代码
    // static bool done = false;
    // if(!done)
    // {
    //     std::cout << "ng1:" << ng1.transpose() << ",ng2:" << ng2.transpose() << std::endl;
    //     std::cout << "R0:" << std::endl << R0 << std::endl;
    //     std::cout << "R0*ng1:" << R0 * ng1 << std::endl;
    //     std::cout << "R0*ng2:" << R0 * ng2 << std::endl;
    // }

    // 我们只想对齐z轴，旋转过程中可能改变了yaw角，再旋转回去
    double yaw = Utility::R2ypr(R0).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;

    // 测试代码
    // if(!done)
    // {
    //     done = true;
    //     std::cout << "yaw:" << yaw << std::endl;
    //     std::cout << "R0*ng1:" << R0 * ng1 << std::endl;
    //     std::cout << "R0*ng2:" << R0 * ng2 << std::endl;
    // }

    return R0;
}
