/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "estimator.h"
#include "../utility/visualization.h"

Estimator::Estimator(): f_manager{Rs}
{
    ROS_INFO("init begins");
    initThreadFlag = false;
    clearState();
}

Estimator::~Estimator()
{
    if (MULTIPLE_THREAD)
    {
        // 等待processThread线程释放
        processThread.join();
        printf("join thread \n");
    }
}

/**
 * 清理状态，缓存数据、变量、滑动窗口数据、位姿等
 * 系统重启或者滑窗优化失败都会调用
*/
void Estimator::clearState()
{
    mProcess.lock();
    while(!accBuf.empty())
        accBuf.pop();
    while(!gyrBuf.empty())
        gyrBuf.pop();
    while(!featureBuf.empty())
        featureBuf.pop();

    prevTime = -1;
    curTime = 0;
    openExEstimation = 0;
    initP = Eigen::Vector3d(0, 0, 0);
    initR = Eigen::Matrix3d::Identity();
    inputImageCnt = 0;
    initFirstPoseFlag = false;

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
        {
            delete pre_integrations[i];
        }
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;

    mProcess.unlock();
}

/**
 * 设置参数
*/
void Estimator::setParameter()
{
    mProcess.lock();
    // 外参，body_T_cam
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
        cout << " exitrinsic cam " << i << endl  << ric[i] << endl << tic[i].transpose() << endl;
    }
    f_manager.setRic(ric);
    ProjectionTwoFrameOneCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTwoFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionOneFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    // 图像的时间戳晚于实际采样时候的时间，硬件传输等因素
    td = TD;
    g = G;
    cout << "set g " << g.transpose() << endl;
    featureTracker.readIntrinsicParameter(CAM_NAMES);

    std::cout << "MULTIPLE_THREAD is " << MULTIPLE_THREAD << '\n';
    if (MULTIPLE_THREAD && !initThreadFlag)
    {
        initThreadFlag = true;
        processThread = std::thread(&Estimator::processMeasurements, this);
    }
    mProcess.unlock();
}

/**
 * 双目，双目+IMU，单目+IMU，如果重新使用IMU，需要重启节点
*/
void Estimator::changeSensorType(int use_imu, int use_stereo)
{
    bool restart = false;
    mProcess.lock();
    if(!use_imu && !use_stereo)
        printf("at least use two sensors! \n");
    else
    {
        if(USE_IMU != use_imu)
        {
            USE_IMU = use_imu;
            if(USE_IMU)
            {
                // reuse imu; restart system
                restart = true;
            }
            else
            {
                if (last_marginalization_info != nullptr)
                    delete last_marginalization_info;

                tmp_pre_integration = nullptr;
                last_marginalization_info = nullptr;
                last_marginalization_parameter_blocks.clear();
            }
        }
        
        STEREO = use_stereo;
        printf("use imu %d use stereo %d\n", USE_IMU, STEREO);
    }
    mProcess.unlock();
    if(restart)
    {
        clearState();
        setParameter();
    }
}

/**
 * 输入一帧图像
 * 1、featureTracker，提取当前帧特征点
 * 2、添加一帧特征点，processMeasurements处理
*/
void Estimator::inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1)
{
    inputImageCnt++;
    // 特征点id，(x,y,z,pu,pv,vx,vy)
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    TicToc featureTrackerTime;

    /**
     * 跟踪一帧图像，提取当前帧特征点
     * 1、用前一帧运动估计特征点在当前帧中的位置，如果特征点没有速度，就直接用前一帧该点位置
     * 2、LK光流跟踪前一帧的特征点，正反向，删除跟丢的点；如果是双目，进行左右匹配，只删右目跟丢的特征点
     * 3、对于前后帧用LK光流跟踪到的匹配特征点，计算基础矩阵，用极线约束进一步剔除outlier点（代码注释掉了）
     * 4、如果特征点不够，剩余的用角点来凑；更新特征点跟踪次数
     * 5、计算特征点归一化相机平面坐标，并计算相对与前一帧移动速度
     * 6、保存当前帧特征点数据（归一化相机平面坐标，像素坐标，归一化相机平面移动速度）
     * 7、展示，左图特征点用颜色区分跟踪次数（红色少，蓝色多），画个箭头指向前一帧特征点位置，如果是双目，右图画个绿色点
    */
    if(_img1.empty())
        featureFrame = featureTracker.trackImage(t, _img);
    else
        featureFrame = featureTracker.trackImage(t, _img, _img1);
    //printf("featureTracker time: %f\n", featureTrackerTime.toc());

    // 发布跟踪图像
    if (SHOW_TRACK)
    {
        cv::Mat imgTrack = featureTracker.getTrackImage();
        pubTrackImage(imgTrack, t);
    }
    
    // 添加一帧特征点，处理
    if(MULTIPLE_THREAD)  
    {     
        if(inputImageCnt % 2 == 0)
        {
            mBuf.lock();
            featureBuf.push(make_pair(t, featureFrame));
            mBuf.unlock();
        }
    }
    else
    {
        mBuf.lock();
        featureBuf.push(make_pair(t, featureFrame));
        mBuf.unlock();
        TicToc processTime;
        processMeasurements();
        printf("process time: %f\n", processTime.toc());
    }
    
}

/**
 * 输入一帧IMU
 * 1、积分预测当前帧状态，latest_time，latest_Q，latest_P，latest_V，latest_acc_0，latest_gyr_0
 * 2、发布里程计
*/
void Estimator::inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity)
{
    mBuf.lock();
    accBuf.push(make_pair(t, linearAcceleration));
    gyrBuf.push(make_pair(t, angularVelocity));
    //printf("input imu with time %f \n", t);
    mBuf.unlock();

    if (solver_flag == NON_LINEAR)
    {
        mPropagate.lock();
        // IMU预测状态，更新QPV
        // latest_time，latest_Q，latest_P，latest_V，latest_acc_0，latest_gyr_0
        fastPredictIMU(t, linearAcceleration, angularVelocity);
        // 发布里程计
        pubLatestOdometry(latest_P, latest_Q, latest_V, t);
        mPropagate.unlock();
    }
}

/**
 * 输入一帧特征点，processMeasurements处理
*/
void Estimator::inputFeature(double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame)
{
    mBuf.lock();
    featureBuf.push(make_pair(t, featureFrame));
    mBuf.unlock();

    if(!MULTIPLE_THREAD)
        processMeasurements();
}

/**
 * 从IMU数据队列中，提取(t0, t1)时间段的数据
*/
bool Estimator::getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector, 
                                vector<pair<double, Eigen::Vector3d>> &gyrVector)
{
    if(accBuf.empty())
    {
        printf("not receive imu\n");
        return false;
    }
    //printf("get imu from %f %f\n", t0, t1);
    //printf("imu fornt time %f   imu end time %f\n", accBuf.front().first, accBuf.back().first);
    if(t1 <= accBuf.back().first)
    {
        while (accBuf.front().first <= t0)
        {
            accBuf.pop();
            gyrBuf.pop();
        }
        while (accBuf.front().first < t1)
        {
            accVector.push_back(accBuf.front());
            accBuf.pop();
            gyrVector.push_back(gyrBuf.front());
            gyrBuf.pop();
        }
        accVector.push_back(accBuf.front());
        gyrVector.push_back(gyrBuf.front());
    }
    else
    {
        printf("wait for imu\n");
        return false;
    }
    return true;
}

/**
 * 已有t时刻的IMU数据
*/
bool Estimator::IMUAvailable(double t)
{
    if(!accBuf.empty() && t <= accBuf.back().first)
        return true;
    else
        return false;
}

/**
 * 处理一帧特征点
*/
void Estimator::processMeasurements()
{
    while (1)
    {
        //printf("process measurments\n");
        pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > feature;
        vector<pair<double, Eigen::Vector3d>> accVector, gyrVector;
        if(!featureBuf.empty())
        {
            // 当前图像帧特征
            feature = featureBuf.front();
            curTime = feature.first + td;
            while(1)
            {
                if ((!USE_IMU  || IMUAvailable(feature.first + td)))
                    break;
                else
                {
                    printf("wait for imu ... \n");
                    if (! MULTIPLE_THREAD)
                        return;
                    std::chrono::milliseconds dura(5);
                    std::this_thread::sleep_for(dura);
                }
            }
            mBuf.lock();
            if(USE_IMU)
                // 前一帧与当前帧图像之间的IMU数据
                getIMUInterval(prevTime, curTime, accVector, gyrVector);

            featureBuf.pop();
            mBuf.unlock();

            if(USE_IMU)
            {
                // 第一帧IMU姿态初始化
                // 用初始时刻加速度方向对齐重力加速度方向，得到一个旋转，使得初始IMU的z轴指向重力加速度方向
                if(!initFirstPoseFlag)
                    initFirstIMUPose(accVector);

                // IMU积分
                // 用前一图像帧位姿，前一图像帧与当前图像帧之间的IMU数据，积分计算得到当前图像帧位姿
                // Rs，Ps，Vs
                for(size_t i = 0; i < accVector.size(); i++)
                {
                    double dt;
                    if(i == 0)
                        dt = accVector[i].first - prevTime;
                    else if (i == accVector.size() - 1)
                        dt = curTime - accVector[i - 1].first;
                    else
                        dt = accVector[i].first - accVector[i - 1].first;
                    processIMU(accVector[i].first, dt, accVector[i].second, gyrVector[i].second);
                }
            }
            mProcess.lock();
            // 处理一帧图像
            processImage(feature.second, feature.first);
            prevTime = curTime;

            printStatistics(*this, 0);

            std_msgs::Header header;
            header.frame_id = "world";
            header.stamp = ros::Time(feature.first);

            // 发布
            pubOdometry(*this, header);
            pubKeyPoses(*this, header);
            pubCameraPose(*this, header);
            pubPointCloud(*this, header);
            pubKeyframe(*this);
            pubTF(*this, header);
            mProcess.unlock();
        }

        if (! MULTIPLE_THREAD)
            break;

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

/**
 * 第一帧IMU姿态初始化
 * 用初始时刻加速度方向对齐重力加速度方向，得到一个旋转，使得初始IMU的z轴指向重力加速度方向
*/
void Estimator::initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector)
{
    printf("init first imu pose\n");
    initFirstPoseFlag = true;
    //return;
    Eigen::Vector3d averAcc(0, 0, 0);
    int n = (int)accVector.size();
    for(size_t i = 0; i < accVector.size(); i++)
    {
        averAcc = averAcc + accVector[i].second;
    }
    // 计算平均加速度
    averAcc = averAcc / n;
    printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());
    // 计算初始IMU的z轴对齐到重力加速度方向所需的旋转，后面每时刻的位姿都是在当前初始IMU坐标系下的，乘上R0就是世界系了
    // 如果初始时刻IMU是绝对水平放置，那么z轴是对应重力加速度方向的，但如果倾斜了，那么就是需要这个旋转让它水平
    Matrix3d R0 = Utility::g2R(averAcc);

    // 打印一下，这里yaw角已经是0了，后面似乎多余
    double yaw = Utility::R2ypr(R0).x();
    // cout << "init R0 before " << endl << R0 << endl;
    // cout << "yaw:" << yaw << endl;
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    Rs[0] = R0;
    cout << "init R0 " << endl << Rs[0] << endl;
    //Vs[0] = Vector3d(5, 0, 0);
}

void Estimator::initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r)
{
    Ps[0] = p;
    Rs[0] = r;
    initP = p;
    initR = r;
}

/**
 * 处理一帧IMU，积分
 * 用前一图像帧位姿，前一图像帧与当前图像帧之间的IMU数据，积分计算得到当前图像帧位姿
 * Rs，Ps，Vs
 * @param t                     当前时刻
 * @param dt                    与前一帧时间间隔
 * @param linear_acceleration   当前时刻加速度
 * @param angular_velocity      当前时刻角速度
*/
void Estimator::processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    // 第一帧
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    if (frame_count != 0)
    {
        // 当前帧预积分器,添加前一图像帧与当前图像帧之间的IMU数据
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        // 缓存IMU数据
        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count;  
        // 前一时刻加速度 
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        // 中值积分，用前一时刻与当前时刻角速度平均值，对时间积分
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        // 当前时刻姿态Q
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        // 当前时刻加速度
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        // 中值积分，用前一时刻与当前时刻加速度平均值，对时间积分
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        // 当前时刻位置P
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        // 当前时刻速度V
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity; 
}

/**
 * 处理一帧图像特征
 * 1、提取前一帧与当前帧的匹配点
 * 2、在线标定外参旋转
 *     利用两帧之间的Camera旋转和IMU积分旋转，构建最小二乘问题，SVD求解外参旋转
 *     1) Camera系，两帧匹配点计算本质矩阵E，分解得到四个解，根据三角化成功点比例确定最终正确解R、t，得到两帧之间的旋转R
 *     2) IMU系，积分计算两帧之间的旋转
 *     3) 根据旋转构建最小二乘问题，SVD求解外参旋转
 * 3、系统初始化
 * 4、3d-2d Pnp求解当前帧位姿
 * 5、三角化当前帧特征点
 * 6、滑窗执行Ceres优化，边缘化，更新滑窗内图像帧的状态（位姿、速度、偏置、外参、逆深度、相机与IMU时差）
 * 7、剔除outlier点
 * 8、用当前帧与前一帧位姿变换，估计下一帧位姿，初始化下一帧特征点的位置
 * 9、移动滑窗，更新特征点的观测帧集合、观测帧索引（在滑窗中的位置）、首帧观测帧和深度值，删除没有观测帧的特征点
 * 10、删除优化后深度值为负的特征点
 * @param image  图像帧特征
 * @param header 时间戳
*/
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double header)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());
    // 添加特征点记录，并检查当前帧是否为关键帧，如果是，marg最早的一帧；否则marg当前帧的前一帧
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
    {
        marginalization_flag = MARGIN_OLD;
        //printf("keyframe\n");
    }
    else
    {
        marginalization_flag = MARGIN_SECOND_NEW;
        //printf("non-keyframe\n");
    }

    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;

    ImageFrame imageframe(image, header);
    // 当前帧预积分（前一帧与当前帧之间的IMU预积分）
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(header, imageframe));
    // 重置预积分器
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    if(ESTIMATE_EXTRINSIC == 2)
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            // 提取前一帧与当前帧的匹配点
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            /**
             * 在线标定外参旋转
             * 利用两帧之间的Camera旋转和IMU积分旋转，构建最小二乘问题，SVD求解外参旋转
             * 1、Camera系，两帧匹配点计算本质矩阵E，分解得到四个解，根据三角化成功点比例确定最终正确解R、t，得到两帧之间的旋转R
             * 2、IMU系，积分计算两帧之间的旋转
             * 3、根据旋转构建最小二乘问题，SVD求解外参旋转
            */
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    // 系统初始化
    if (solver_flag == INITIAL)
    {
        // monocular + IMU initilization
        // 单目+IMU系统初始化
        if (!STEREO && USE_IMU)
        {
            // 要求滑窗满
            if (frame_count == WINDOW_SIZE)
            {
                bool result = false;
                // 如果上次初始化没有成功，要求间隔0.1s
                if(ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1)
                {
                    /**
                     * todo
                     * 系统初始化
                     * 1、计算滑窗内IMU加速度的标准差，用于判断移动快慢
                     * 2、在滑窗中找到与当前帧具有足够大的视差，同时匹配较为准确的一帧，计算相对位姿变换
                     *   1) 提取滑窗中每帧与当前帧之间的匹配点（要求点在两帧之间一直被跟踪到，属于稳定共视点），超过20个则计算视差
                     *   2) 两帧匹配点计算本质矩阵E，恢复R、t
                     *   3) 视差超过30像素，匹配内点数超过12个，则认为符合要求，返回当前帧
                     * 3、以上面找到的这一帧为参考系，Pnp计算滑窗每帧位姿，然后三角化所有特征点，构建BA（最小化点三角化前后误差）优化每帧位姿
                     *   1) 3d-2d Pnp求解每帧位姿
                     *   2) 对每帧与l帧、当前帧三角化
                     *   3) 构建BA，最小化点三角化前后误差，优化每帧位姿
                     *   4) 保存三角化点
                     * 4、对滑窗中所有帧执行Pnp优化位姿
                     * 5、Camera与IMU初始化，零偏、尺度、重力方向
                    */
                    result = initialStructure();
                    initial_timestamp = header;   
                }
                if(result)
                {
                    // 滑窗执行Ceres优化，边缘化，更新滑窗内图像帧的状态（位姿、速度、偏置、外参、逆深度、相机与IMU时差）
                    optimization();
                    // 用优化后的当前帧位姿更新IMU积分的基础位姿，用于展示IMU轨迹
                    updateLatestStates();
                    solver_flag = NON_LINEAR;
                    // 移动滑窗，更新特征点的观测帧集合、观测帧索引（在滑窗中的位置）、首帧观测帧和深度值，删除没有观测帧的特征点
                    slideWindow();
                    ROS_INFO("Initialization finish!");
                }
                else
                    // 移动滑窗，更新特征点的观测帧集合、观测帧索引（在滑窗中的位置）、首帧观测帧和深度值，删除没有观测帧的特征点
                    slideWindow();
            }
        }

        // stereo + IMU initilization
        // 双目+IMU系统初始化
        if(STEREO && USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            if (frame_count == WINDOW_SIZE)
            {
                map<double, ImageFrame>::iterator frame_it;
                int i = 0;
                for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++)
                {
                    frame_it->second.R = Rs[i];
                    frame_it->second.T = Ps[i];
                    i++;
                }
                // 零偏初始化
                solveGyroscopeBias(all_image_frame, Bgs);
                for (int i = 0; i <= WINDOW_SIZE; i++)
                {
                    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
                }
                // 滑窗执行Ceres优化，边缘化，更新滑窗内图像帧的状态（位姿、速度、偏置、外参、逆深度、相机与IMU时差）
                optimization();
                // 用优化后的当前帧位姿更新IMU积分的基础位姿，用于展示IMU轨迹
                updateLatestStates();
                solver_flag = NON_LINEAR;
                // 移动滑窗，更新特征点的观测帧集合、观测帧索引（在滑窗中的位置）、首帧观测帧和深度值，删除没有观测帧的特征点
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }

        // stereo only initilization
        // 双目系统初始化
        if(STEREO && !USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            optimization();

            if(frame_count == WINDOW_SIZE)
            {
                // 滑窗执行Ceres优化，边缘化，更新滑窗内图像帧的状态（位姿、速度、偏置、外参、逆深度、相机与IMU时差）
                optimization();
                // 用优化后的当前帧位姿更新IMU积分的基础位姿，用于展示IMU轨迹
                updateLatestStates();
                solver_flag = NON_LINEAR;
                // 移动滑窗，更新特征点的观测帧集合、观测帧索引（在滑窗中的位置）、首帧观测帧和深度值，删除没有观测帧的特征点
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }

        if(frame_count < WINDOW_SIZE)
        {
            frame_count++;
            int prev_frame = frame_count - 1;
            Ps[frame_count] = Ps[prev_frame];
            Vs[frame_count] = Vs[prev_frame];
            Rs[frame_count] = Rs[prev_frame];
            Bas[frame_count] = Bas[prev_frame];
            Bgs[frame_count] = Bgs[prev_frame];
        }

    }
    else
    {
        TicToc t_solve;
        // 3d-2d Pnp求解当前帧位姿
        if(!USE_IMU)
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
        // 三角化当前帧特征点
        f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
        // 滑窗执行Ceres优化，边缘化，更新滑窗内图像帧的状态（位姿、速度、偏置、外参、逆深度、相机与IMU时差）
        optimization();
        set<int> removeIndex;
        /**
         * 剔除outlier点
         * 遍历特征点，计算观测帧与首帧观测帧之间的重投影误差，计算误差均值，超过3个像素则被剔除
        */
        outliersRejection(removeIndex);
        // 实际调用剔除
        f_manager.removeOutlier(removeIndex);
        if (! MULTIPLE_THREAD)
        {
            featureTracker.removeOutliers(removeIndex);
            // 用当前帧与前一帧位姿变换，估计下一帧位姿，初始化下一帧特征点的位置
            predictPtsInNextFrame();
        }
            
        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        // 失败检测，失败了重置系统
        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }

        // 移动滑窗，更新特征点的观测帧集合、观测帧索引（在滑窗中的位置）、首帧观测帧和深度值，删除没有观测帧的特征点
        slideWindow();
        // 删除优化后深度值为负的特征点
        f_manager.removeFailures();
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
        // 用优化后的当前帧位姿更新IMU积分的基础位姿，用于展示IMU轨迹
        updateLatestStates();
    }  
}

/**
 * todo
 * 系统初始化
 * 1、计算滑窗内IMU加速度的标准差，用于判断移动快慢
 * 2、在滑窗中找到与当前帧具有足够大的视差，同时匹配较为准确的一帧，计算相对位姿变换
 *   1) 提取滑窗中每帧与当前帧之间的匹配点（要求点在两帧之间一直被跟踪到，属于稳定共视点），超过20个则计算视差
 *   2) 两帧匹配点计算本质矩阵E，恢复R、t
 *   3) 视差超过30像素，匹配内点数超过12个，则认为符合要求，返回当前帧
 * 3、以上面找到的这一帧为参考系，Pnp计算滑窗每帧位姿，然后三角化所有特征点，构建BA（最小化点三角化前后误差）优化每帧位姿
 *   1) 3d-2d Pnp求解每帧位姿
 *   2) 对每帧与l帧、当前帧三角化
 *   3) 构建BA，最小化点三角化前后误差，优化每帧位姿
 *   4) 保存三角化点
 * 4、对滑窗中所有帧执行Pnp优化位姿
 * 5、Camera与IMU初始化，零偏、尺度、重力方向
*/
bool Estimator::initialStructure()
{
    TicToc t_sfm;
    //check imu observibility
    // 计算滑窗内IMU加速度的标准差，用于判断移动快慢，为什么不用旋转或者平移量判断 todo
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        // 从第2帧开始累加每帧加速度
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        // 加速度均值
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        // 加速度标准差
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        //ROS_WARN("IMU variation %f!", var);
        if(var < 0.25)
        {
            ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }
    // global sfm
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    // 遍历当前帧特征点
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        // 遍历特征点出现的帧
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            // 特征点归一化相机平面点
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    } 
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    /**
     * 在滑窗中找到与当前帧具有足够大的视差，同时匹配较为准确的一帧，计算相对位姿变换
     * 1、提取滑窗中每帧与当前帧之间的匹配点（要求点在两帧之间一直被跟踪到，属于稳定共视点），超过20个则计算视差
     * 2、两帧匹配点计算本质矩阵E，恢复R、t
     * 3、视差超过30像素，匹配内点数超过12个，则认为符合要求，返回当前帧
    */
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    /**
     * 以第l帧为参考系，Pnp计算滑窗每帧位姿，然后三角化所有特征点，构建BA（最小化点三角化前后误差）优化每帧位姿
     * 1、3d-2d Pnp求解每帧位姿
     * 2、对每帧与l帧、当前帧三角化
     * 3、构建BA，最小化点三角化前后误差，优化每帧位姿
     * 4、保存三角化点
    */
    GlobalSFM sfm;
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        // 如果SFM三角化失败，marg最早的一帧
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame
    // 对滑窗中所有帧执行Pnp位姿优化 todo
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin( );
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if((frame_it->first) == Headers[i])
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if((frame_it->first) > Headers[i])
        {
            i++;
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        // 遍历当前帧特征点
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            // 遍历特征点的观测帧，提取像素坐标
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    // 特征点3d坐标，sfm三角化后的点
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    // 特征点像素坐标
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        // 3d-2d Pnp求解位姿
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }
    // Camera与IMU初始化，零偏、尺度、重力方向
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }

}

/**
 * todo
*/
bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    //solve scale
    // Camera与IMU初始化，零偏、尺度、重力方向
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i]].R;
        Vector3d Pi = all_image_frame[Headers[i]].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i]].is_key_frame = true;
    }

    double s = (x.tail<1>())(0);
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 

    f_manager.clearDepth();
    f_manager.triangulate(frame_count, Ps, Rs, tic, ric);

    return true;
}

/**
 * 在滑窗中找到与当前帧具有足够大的视差，同时匹配较为准确的一帧，计算相对位姿变换
 * 1、提取滑窗中每帧与当前帧之间的匹配点（要求点在两帧之间一直被跟踪到，属于稳定共视点），超过20个则计算视差
 * 2、两帧匹配点计算本质矩阵E，恢复R、t
 * 3、视差超过30像素，匹配内点数超过12个，则认为符合要求，返回当前帧
*/
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contains enough correspondance and parallex with newest frame
    // 遍历滑窗
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        // 提取滑窗中每帧与当前帧之间的匹配点（要求点在两帧之间一直被跟踪到，属于稳定共视点）
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        // 匹配点超过20个
        if (corres.size() > 20)
        {
            // 计算匹配点之间的累积、平均视差（归一化相机坐标系下），作为当前两帧之间的视差
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            // 视差超过30个像素点；两帧匹配点计算本质矩阵E，恢复R、t，内点数超过12个
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

/**
 * 滑窗中的帧位姿、速度、偏置、外参、特征点逆深度等参数，转换成数组
 * ceres参数需要是数组形式
*/
void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        // 平移、旋转
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        if(USE_IMU)
        {
            // 速度、加速度偏置、角速度偏置
            para_SpeedBias[i][0] = Vs[i].x();
            para_SpeedBias[i][1] = Vs[i].y();
            para_SpeedBias[i][2] = Vs[i].z();

            para_SpeedBias[i][3] = Bas[i].x();
            para_SpeedBias[i][4] = Bas[i].y();
            para_SpeedBias[i][5] = Bas[i].z();

            para_SpeedBias[i][6] = Bgs[i].x();
            para_SpeedBias[i][7] = Bgs[i].y();
            para_SpeedBias[i][8] = Bgs[i].z();
        }
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        // 相机与IMU外参
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    // 当前帧特征点逆深度，限观测帧数量大于4个的特征点
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);

    // 相机与IMU时差
    para_Td[0][0] = td;
}

/**
 * 更新优化后的参数，包括位姿、速度、偏置、外参、特征点逆深度、相机与IMU时差
*/
void Estimator::double2vector()
{
    // 第一帧优化前位姿
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    // 如果上一次优化失败了，Rs、Ps都会被清空，用备份的last_R0、last_P0
    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }

    // 使用IMU时，第一帧没有固定，会有姿态变化
    if(USE_IMU)
    {
        // 本次优化后第一帧位姿
        Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                          para_Pose[0][3],
                                                          para_Pose[0][4],
                                                          para_Pose[0][5]).toRotationMatrix());
        // yaw角差量
        double y_diff = origin_R0.x() - origin_R00.x();
        // yaw角差量对应旋转矩阵
        Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));

        // pitch角接近90°，todo
        if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
        {
            ROS_DEBUG("euler singular point!");
            // 计算旋转位姿变换
            rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                           para_Pose[0][3],
                                           para_Pose[0][4],
                                           para_Pose[0][5]).toRotationMatrix().transpose();
        }

        // 遍历滑窗，位姿、速度全部施加优化前后第一帧的位姿变换（只有旋转） todo
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {

            Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
            
            Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                    para_Pose[i][1] - para_Pose[0][1],
                                    para_Pose[i][2] - para_Pose[0][2]) + origin_P0;

            Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                        para_SpeedBias[i][1],
                                        para_SpeedBias[i][2]);

            Bas[i] = Vector3d(para_SpeedBias[i][3],
                                para_SpeedBias[i][4],
                                para_SpeedBias[i][5]);

            Bgs[i] = Vector3d(para_SpeedBias[i][6],
                                para_SpeedBias[i][7],
                                para_SpeedBias[i][8]);
            
        }
    }
    // 不使用IMU时，第一帧固定，后面的位姿直接赋值
    else
    {
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
            
            Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
        }
    }

    // 更新外参
    if(USE_IMU)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            tic[i] = Vector3d(para_Ex_Pose[i][0],
                              para_Ex_Pose[i][1],
                              para_Ex_Pose[i][2]);
            ric[i] = Quaterniond(para_Ex_Pose[i][6],
                                 para_Ex_Pose[i][3],
                                 para_Ex_Pose[i][4],
                                 para_Ex_Pose[i][5]).normalized().toRotationMatrix();
        }
    }

    // 更新逆深度
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);

    // 更新相机与IMU时差
    if(USE_IMU)
        td = para_Td[0][0];

}

/**
 * 失败检测
*/
bool Estimator::failureDetection()
{
    return false;
    // 上一帧老特征点几乎全丢
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    // 偏置太大
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    // 偏置太大
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    // 优化前后当前帧位置相差过大
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        //ROS_INFO(" big translation");
        //return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        //ROS_INFO(" big z translation");
        //return true; 
    }
    // 当前帧位姿优化前后角度相差过大
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}

/**
 * 滑窗执行Ceres优化，边缘化，更新滑窗内图像帧的状态（位姿、速度、偏置、外参、逆深度、相机与IMU时差）
*/
void Estimator::optimization()
{
    TicToc t_whole, t_prepare;
    // 滑窗中的帧位姿、速度、偏置、外参、特征点逆深度等参数，转换成数组
    vector2double();

    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = NULL;
    loss_function = new ceres::HuberLoss(1.0);
    //loss_function = new ceres::CauchyLoss(1.0 / FOCAL_LENGTH);
    //ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
    
    /**
     * Step1. 调用AddParameterBlock，显式添加待优化变量（类似于g2o中添加顶点），需要固定的顶点固定一下
    */

    // 遍历滑窗，添加位姿、速度、偏置参数
    for (int i = 0; i < frame_count + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        if(USE_IMU)
            problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    // 如果不使用IMU，固定第一帧位姿，IMU下第一帧不固定
    if(!USE_IMU)
        problem.SetParameterBlockConstant(para_Pose[0]);

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        // 添加相机与IMU外参
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        // 估计外参
        if ((ESTIMATE_EXTRINSIC && frame_count == WINDOW_SIZE && Vs[0].norm() > 0.2) || openExEstimation)
        {
            //ROS_INFO("estimate extinsic param");
            openExEstimation = 1;
        }
        else
        {
            //ROS_INFO("fix extinsic param");
            // 不估计外参的时候，固定外参
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
    }
    // 添加相机与IMU时差
    problem.AddParameterBlock(para_Td[0], 1);

    if (!ESTIMATE_TD || Vs[0].norm() < 0.2)
        problem.SetParameterBlockConstant(para_Td[0]);

    /**
     * Step2. 调用AddResidualBlock，添加各种残差数据（类似于g2o中的边）
    */

    // (1) 添加先验残差，通过Marg的舒尔补操作，将被Marg部分的信息叠加到了保留变量的信息上
    if (last_marginalization_info && last_marginalization_info->valid)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }
    // (2) 添加IMU残差
    if(USE_IMU)
    {
        for (int i = 0; i < frame_count; i++)
        {
            int j = i + 1;
            if (pre_integrations[j]->sum_dt > 10.0)
                continue;
            // 前后帧之间建立IMU残差
            IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
            // 后面四个参数为变量初始值，优化过程中会更新
            problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
        }
    }

    // (3) 添加视觉重投影残差
    int f_m_cnt = 0;
    int feature_index = -1;
    // 遍历特征点
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;
 
        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        
        // 首帧归一化相机平面点
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        // 遍历特征点的观测帧
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            // 非首帧观测帧
            if (imu_i != imu_j)
            {
                // 当前观测帧归一化相机平面点
                Vector3d pts_j = it_per_frame.point;
                // 首帧与当前观测帧建立重投影误差
                ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                // 优化变量：首帧位姿，当前帧位姿，外参（左目），特征点逆深度，相机与IMU时差
                problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
            }

            // 双目，重投影误差
            if(STEREO && it_per_frame.is_stereo)
            {                
                Vector3d pts_j_right = it_per_frame.pointRight;
                if(imu_i != imu_j)
                {
                    // 首帧与当前观测帧右目建立重投影误差
                    ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    // 优化变量：首帧位姿，当前帧位姿，外参（左目），外参（右目），特征点逆深度，相机与IMU时差
                    problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
                else
                {
                    // 首帧左右目建立重投影误差
                    ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    // 优化变量：外参（左目），外参（右目），特征点逆深度，相机与IMU时差
                    problem.AddResidualBlock(f, loss_function, para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
               
            }
            f_m_cnt++;
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    //printf("prepare for ceres: %f \n", t_prepare.toc());

    /**
     * Step3. 设置优化器参数，执行优化
    */

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    //printf("solver costs: %f \n", t_solver.toc());

    // 更新优化后的参数，包括位姿、速度、偏置、外参、特征点逆深度、相机与IMU时差
    double2vector();
    //printf("frame_count: %d \n", frame_count);

    if(frame_count < WINDOW_SIZE)
        return;
    
    /**
     * Step4. 边缘化操作
    */

    // 以下是边缘化操作
    TicToc t_whole_marginalization;
    // Marg最早帧
    if (marginalization_flag == MARGIN_OLD)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        // 滑窗中的帧位姿、速度、偏置、外参、特征点逆深度等参数，转换成数组
        vector2double();

        // 先验残差
        if (last_marginalization_info && last_marginalization_info->valid)
        {
            vector<int> drop_set;
            // 上一次Marg剩下的参数块
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        // 滑窗首帧与后一帧之间的IMU残差 
        if(USE_IMU)
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        // 滑窗首帧与其他帧之间的视觉重投影残差
        {
            // 遍历特征点
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (it_per_id.used_num < 4)
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                // 首帧观测帧归一化相机平面点
                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                // 遍历观测帧
                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    // 非首个观测帧
                    if(imu_i != imu_j)
                    {
                        Vector3d pts_j = it_per_frame.point;
                        ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    if(STEREO && it_per_frame.is_stereo)
                    {
                        Vector3d pts_j_right = it_per_frame.pointRight;
                        if(imu_i != imu_j)
                        {
                            ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                           vector<int>{0, 4});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                        else
                        {
                            ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                           vector<int>{2});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                    }
                }
            }
        }

        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
        
        // 执行marg边缘化
        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        // marg首帧之后，将参数数组中每个位置的值设为前面元素的值，记录到addr_shift里面
        // [<p1,p0>,<p2,p1>,...]
        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            printf("pose %ld,%ld\n",reinterpret_cast<long>(para_Pose[i]),reinterpret_cast<long>(para_Pose[i-1]));
            if(USE_IMU)
            {
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                printf("speedBias %ld,%ld\n",reinterpret_cast<long>(para_SpeedBias[i]),reinterpret_cast<long>(para_SpeedBias[i-1]));
            }
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            printf("exPose %ld,%ld\n",reinterpret_cast<long>(para_Ex_Pose[i]),reinterpret_cast<long>(para_Ex_Pose[i]));
        }

        addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        printf("td %ld,%ld\n",reinterpret_cast<long>(para_Td[0]),reinterpret_cast<long>(para_Td[0]));

        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        // 保存marg信息
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
        
    }
    // Marg新帧
    else
    {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info && last_marginalization_info->valid)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            
            // [<p0,p0>, <p1,p1>,...,<pn-1,pn-1>,<pn,pn-1>]
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                // 被Marg帧
                if (i == WINDOW_SIZE - 1)
                    continue;
                // 当前帧
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

            
            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
            
        }
    }
    //printf("whole marginalization costs: %f \n", t_whole_marginalization.toc());
    //printf("whole time for ceres: %f \n", t_whole.toc());
}

/**
 * 移动滑窗，更新特征点的观测帧集合、观测帧索引（在滑窗中的位置）、首帧观测帧和深度值，删除没有观测帧的特征点
*/
void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0];
        // 备份
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            // 删除第一帧，全部数据往前移
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Headers[i] = Headers[i + 1];
                Rs[i].swap(Rs[i + 1]);
                Ps[i].swap(Ps[i + 1]);
                if(USE_IMU)
                {
                    std::swap(pre_integrations[i], pre_integrations[i + 1]);

                    dt_buf[i].swap(dt_buf[i + 1]);
                    linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                    angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                    Vs[i].swap(Vs[i + 1]);
                    Bas[i].swap(Bas[i + 1]);
                    Bgs[i].swap(Bgs[i + 1]);
                }
            }
            // 新增末尾帧，用当前帧数据初始化
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

            if(USE_IMU)
            {
                Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
                Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
                Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
            // 删除首帧image
            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                all_image_frame.erase(all_image_frame.begin(), it_0);
            }
            // 移动滑窗，从特征点观测帧集合中删除该帧，计算新首帧深度值
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            // 当前帧的前一帧用当前帧数据代替，当前帧还留下了
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];

            if(USE_IMU)
            {
                // 当前帧的imu数据
                for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
                {
                    double tmp_dt = dt_buf[frame_count][i];
                    Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                    Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                    pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                    dt_buf[frame_count - 1].push_back(tmp_dt);
                    linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                    angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
                }

                Vs[frame_count - 1] = Vs[frame_count];
                Bas[frame_count - 1] = Bas[frame_count];
                Bgs[frame_count - 1] = Bgs[frame_count];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
            // 边缘化当前帧前面一帧后，从特征点的观测帧集合中删除该帧，如果特征点没有观测帧了，删除这个特征点
            slideWindowNew();
        }
    }
}

/**
 * 边缘化当前帧前面一帧后，从特征点的观测帧集合中删除该帧，如果特征点没有观测帧了，删除这个特征点
*/
void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}

/**
 * 移动滑窗，从特征点观测帧集合中删除该帧，计算新首帧深度值
*/
void Estimator::slideWindowOld()
{
    sum_of_back++;

    // NON_LINEAR表示已经初始化过了
    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        // marg帧的位姿 Rwc
        R0 = back_R0 * ric[0];
        // 后面一帧的位姿 Rwc
        R1 = Rs[0] * ric[0];
        // marg帧的位置
        P0 = back_P0 + back_R0 * tic[0];
        // 后面一帧的位置
        P1 = Ps[0] + Rs[0] * tic[0];
        /**
         * 边缘化第一帧后，从特征点的观测帧集合中删除该帧，观测帧的索引相应全部-1，如果特征点没有观测帧少于2帧，删除这个特征点
         * 与首帧绑定的estimated_depth深度值，重新计算
        */
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        // 边缘化第一帧后，从特征点的观测帧集合中删除该帧，观测帧的索引相应全部-1，如果特征点没有观测帧了，删除这个特征点
        f_manager.removeBack();
}

/**
 * 当前帧位姿 Twi
*/
void Estimator::getPoseInWorldFrame(Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[frame_count];
    T.block<3, 1>(0, 3) = Ps[frame_count];
}

/**
 * 位姿 Twi
*/
void Estimator::getPoseInWorldFrame(int index, Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[index];
    T.block<3, 1>(0, 3) = Ps[index];
}

/**
 * 用当前帧与前一帧位姿变换，估计下一帧位姿，初始化下一帧特征点的位置
*/
void Estimator::predictPtsInNextFrame()
{
    //printf("predict pts in next frame\n");
    if(frame_count < 2)
        return;
    // predict next pose. Assume constant velocity motion
    // 当前帧位姿、前一帧位姿Twi
    Eigen::Matrix4d curT, prevT, nextT;
    getPoseInWorldFrame(curT);
    getPoseInWorldFrame(frame_count - 1, prevT);
    // 用前一帧位姿与当前帧位姿的变换，预测下一帧位姿
    nextT = curT * (prevT.inverse() * curT);
    map<int, Eigen::Vector3d> predictPts;

    // 遍历特征点
    for (auto &it_per_id : f_manager.feature)
    {
        if(it_per_id.estimated_depth > 0)
        {
            int firstIndex = it_per_id.start_frame;
            int lastIndex = it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1;
            //printf("cur frame index  %d last frame index %d\n", frame_count, lastIndex);
            if((int)it_per_id.feature_per_frame.size() >= 2 && lastIndex == frame_count)
            {
                double depth = it_per_id.estimated_depth;
                // 特征点在首帧观测帧中的IMU系坐标
                Vector3d pts_j = ric[0] * (depth * it_per_id.feature_per_frame[0].point) + tic[0];
                // 世界坐标
                Vector3d pts_w = Rs[firstIndex] * pts_j + Ps[firstIndex];
                // 转换到下一帧IMU坐标系
                Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3));
                // 转到相机系
                Vector3d pts_cam = ric[0].transpose() * (pts_local - tic[0]);
                int ptsIndex = it_per_id.feature_id;
                predictPts[ptsIndex] = pts_cam;
            }
        }
    }
    // 设置下一帧跟踪点初始位置
    featureTracker.setPrediction(predictPts);
    //printf("estimator output %d predict pts\n",(int)predictPts.size());
}

/**
 * 计算重投影误差
*/
double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                 Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, 
                                 double depth, Vector3d &uvi, Vector3d &uvj)
{
    // i点对应世界坐标
    Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    // 转换到j相机坐标系
    Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    // 在j归一化相机坐标系下计算误差
    Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}

/**
 * 剔除outlier点
 * 遍历特征点，计算观测帧与首帧观测帧之间的重投影误差，计算误差均值，超过3个像素则被剔除
*/
void Estimator::outliersRejection(set<int> &removeIndex)
{
    //return;
    int feature_index = -1;
    // 遍历特征点
    for (auto &it_per_id : f_manager.feature)
    {
        double err = 0;
        int errCnt = 0;
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;

        feature_index ++;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        // 首帧观测帧的归一化相机坐标点
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        // 深度值
        double depth = it_per_id.estimated_depth;
        // 遍历观测帧，计算与首帧观测帧之间的特征点重投影误差
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            // 非首帧观测帧
            if (imu_i != imu_j)
            {
                // 计算重投影误差
                Vector3d pts_j = it_per_frame.point;             
                double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                                                    Rs[imu_j], Ps[imu_j], ric[0], tic[0],
                                                    depth, pts_i, pts_j);
                err += tmp_error;
                errCnt++;
                //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
            }
            // need to rewrite projecton factor.........
            // 双目，右目同样计算一次
            if(STEREO && it_per_frame.is_stereo)
            {
                
                Vector3d pts_j_right = it_per_frame.pointRight;
                if(imu_i != imu_j)
                {            
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                                                        Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                        depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
                else
                {
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                                                        Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                        depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }       
            }
        }
        // 重投影误差均值（归一化相机坐标系）
        double ave_err = err / errCnt;
        // 误差超过3个像素，就不要了
        if(ave_err * FOCAL_LENGTH > 3)
            removeIndex.insert(it_per_id.feature_id);

    }
}

/**
 * IMU预测状态，更新QPV
 * latest_time，latest_Q，latest_P，latest_V，latest_acc_0，latest_gyr_0
*/
void Estimator::fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity)
{
    double dt = t - latest_time;
    latest_time = t;
    // 前一帧加速度（世界系）
    Eigen::Vector3d un_acc_0 = latest_Q * (latest_acc_0 - latest_Ba) - g;
    Eigen::Vector3d un_gyr = 0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg;
    // 更新旋转Q
    latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt);
    Eigen::Vector3d un_acc_1 = latest_Q * (linear_acceleration - latest_Ba) - g;
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    // 更新位置P
    latest_P = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
    // 更新速度V
    latest_V = latest_V + dt * un_acc;
    latest_acc_0 = linear_acceleration;
    latest_gyr_0 = angular_velocity;
}

/**
 * 用优化后的当前帧位姿更新IMU积分的基础位姿，用于展示IMU轨迹
*/
void Estimator::updateLatestStates()
{
    mPropagate.lock();
    // 更新当前帧状态，是接下来IMU积分的基础
    latest_time = Headers[frame_count] + td;
    latest_P = Ps[frame_count];
    latest_Q = Rs[frame_count];
    latest_V = Vs[frame_count];
    latest_Ba = Bas[frame_count];
    latest_Bg = Bgs[frame_count];
    latest_acc_0 = acc_0;
    latest_gyr_0 = gyr_0;
    mBuf.lock();
    queue<pair<double, Eigen::Vector3d>> tmp_accBuf = accBuf;
    queue<pair<double, Eigen::Vector3d>> tmp_gyrBuf = gyrBuf;
    mBuf.unlock();
    while(!tmp_accBuf.empty())
    {
        // 当前帧之后的一部分imu数据，用当前帧位姿预测，更新到最新，这个只用于IMU路径的展示
        double t = tmp_accBuf.front().first;
        Eigen::Vector3d acc = tmp_accBuf.front().second;
        Eigen::Vector3d gyr = tmp_gyrBuf.front().second;
        fastPredictIMU(t, acc, gyr);
        tmp_accBuf.pop();
        tmp_gyrBuf.pop();
    }
    mPropagate.unlock();
}
