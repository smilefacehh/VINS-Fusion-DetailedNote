/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "marginalization_factor.h"

/**
 * 执行一次优化，得到优化后变量、残差
*/
void ResidualBlockInfo::Evaluate()
{
    // 残差维度
    residuals.resize(cost_function->num_residuals());
    // 每个变量块中变量个数
    std::vector<int> block_sizes = cost_function->parameter_block_sizes();
    raw_jacobians = new double *[block_sizes.size()];
    jacobians.resize(block_sizes.size());

    for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
    {
        jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]);
        raw_jacobians[i] = jacobians[i].data();
        //dim += block_sizes[i] == 7 ? 6 : block_sizes[i];
    }
    // 执行一次优化，得到残差、Jacobian。注：因为Marg操作是在优化结束之后执行的，当前变量基本上已经是最优的了，这里再执行一次，残差、Jacobian都很小
    cost_function->Evaluate(parameter_blocks.data(), residuals.data(), raw_jacobians);

    //std::vector<int> tmp_idx(block_sizes.size());
    //Eigen::MatrixXd tmp(dim, dim);
    //for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
    //{
    //    int size_i = localSize(block_sizes[i]);
    //    Eigen::MatrixXd jacobian_i = jacobians[i].leftCols(size_i);
    //    for (int j = 0, sub_idx = 0; j < static_cast<int>(parameter_blocks.size()); sub_idx += block_sizes[j] == 7 ? 6 : block_sizes[j], j++)
    //    {
    //        int size_j = localSize(block_sizes[j]);
    //        Eigen::MatrixXd jacobian_j = jacobians[j].leftCols(size_j);
    //        tmp_idx[j] = sub_idx;
    //        tmp.block(tmp_idx[i], tmp_idx[j], size_i, size_j) = jacobian_i.transpose() * jacobian_j;
    //    }
    //}
    //Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(tmp);
    //std::cout << saes.eigenvalues() << std::endl;
    //ROS_ASSERT(saes.eigenvalues().minCoeff() >= -1e-6);

    if (loss_function)
    {
        double residual_scaling_, alpha_sq_norm_;

        double sq_norm, rho[3];

        sq_norm = residuals.squaredNorm();
        loss_function->Evaluate(sq_norm, rho);
        //printf("sq_norm: %f, rho[0]: %f, rho[1]: %f, rho[2]: %f\n", sq_norm, rho[0], rho[1], rho[2]);

        double sqrt_rho1_ = sqrt(rho[1]);

        if ((sq_norm == 0.0) || (rho[2] <= 0.0))
        {
            residual_scaling_ = sqrt_rho1_;
            alpha_sq_norm_ = 0.0;
        }
        else
        {
            const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
            const double alpha = 1.0 - sqrt(D);
            residual_scaling_ = sqrt_rho1_ / (1 - alpha);
            alpha_sq_norm_ = alpha / sq_norm;
        }

        for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
        {
            jacobians[i] = sqrt_rho1_ * (jacobians[i] - alpha_sq_norm_ * residuals * (residuals.transpose() * jacobians[i]));
        }

        residuals *= residual_scaling_;
    }
}

MarginalizationInfo::~MarginalizationInfo()
{
    //ROS_WARN("release marginlizationinfo");
    
    for (auto it = parameter_block_data.begin(); it != parameter_block_data.end(); ++it)
        delete it->second;

    for (int i = 0; i < (int)factors.size(); i++)
    {

        delete[] factors[i]->raw_jacobians;
        
        delete factors[i]->cost_function;

        delete factors[i];
    }
}

/**
 * 添加与当前Marg帧有关联的残差块信息，后面用于计算Jacobian、残差、变量值
 * 1、factors 所有残差块
 *    1) 前一帧Marg留下的先验
 *    2) 当前Marg帧与后一帧IMU残差
 *    3) 当前Marg帧与滑窗内其他帧的视觉残差
 * 2、parameter_block_size <变量块起始地址, 变量块尺寸>
 * 3、parameter_block_idx <将要被丢弃的变量块起始地址，0>
*/
void MarginalizationInfo::addResidualBlockInfo(ResidualBlockInfo *residual_block_info)
{
    // 添加残差项
    factors.emplace_back(residual_block_info);

    // 变量块，比如平移和旋转四元数加在一起，构成一个pose变量块
    std::vector<double *> &parameter_blocks = residual_block_info->parameter_blocks;
    // 每个变量块的尺寸
    std::vector<int> parameter_block_sizes = residual_block_info->cost_function->parameter_block_sizes();

    // 遍历每个变量块
    for (int i = 0; i < static_cast<int>(residual_block_info->parameter_blocks.size()); i++)
    {
        // 变量块起始地址
        double *addr = parameter_blocks[i];
        // 变量块中参数的个数，记为尺寸
        int size = parameter_block_sizes[i];
        // 记录 <变量块起始地址, 变量块尺寸>
        parameter_block_size[reinterpret_cast<long>(addr)] = size;
    }

    // 遍历将要被丢弃的变量块
    for (int i = 0; i < static_cast<int>(residual_block_info->drop_set.size()); i++)
    {
        // 将要被丢弃的变量块起始地址
        double *addr = parameter_blocks[residual_block_info->drop_set[i]];
        parameter_block_idx[reinterpret_cast<long>(addr)] = 0;
    }
}

/**
 * 执行一次优化，得到变量数据、Jacobian、残差，后面用于构造H、b（normal equation），然后对H执行舒尔补操作，将Marg变量的信息转嫁到与之相关联的变量上
 * 1、执行一次优化，得到优化后变量parameter_block_data
 * 2、更新Jacobian、残差
*/
void MarginalizationInfo::preMarginalize()
{
    // 遍历当前Marg帧关联的所有残差块
    for (auto it : factors)
    {
        // 执行一次优化，得到优化后变量、残差
        it->Evaluate();

        // 遍历优化后变量，拷贝一份到parameter_block_data
        std::vector<int> block_sizes = it->cost_function->parameter_block_sizes();
        for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
        {
            // 变量块起始地址
            long addr = reinterpret_cast<long>(it->parameter_blocks[i]);
            int size = block_sizes[i];
            if (parameter_block_data.find(addr) == parameter_block_data.end())
            {
                double *data = new double[size];
                memcpy(data, it->parameter_blocks[i], sizeof(double) * size);
                parameter_block_data[addr] = data;
            }
        }
    }
}

int MarginalizationInfo::localSize(int size) const
{
    return size == 7 ? 6 : size;
}

int MarginalizationInfo::globalSize(int size) const
{
    return size == 6 ? 7 : size;
}

/**
 * 构造信息矩阵A = J^T·J， b = J^T·r
*/
void* ThreadsConstructA(void* threadsstruct)
{
    ThreadsStruct* p = ((ThreadsStruct*)threadsstruct);
    // 遍历残差块
    for (auto it : p->sub_factors)
    {
        // 遍历变量块
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
            // 变量块索引
            int idx_i = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            // 变量块长度
            int size_i = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])];
            if (size_i == 7)
                size_i = 6;
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                int idx_j = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])];
                if (size_j == 7)
                    size_j = 6;
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                if (i == j)
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else
                {
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    p->A.block(idx_j, idx_i, size_j, size_i) = p->A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            p->b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    return threadsstruct;
}

/**
 * 边缘化操作
 * 1、需要Marg的变量放前面，计算相关索引
 * 2、从Jacobian和Residual构造信息矩阵A = J^T·J， b = J^T·r，这个A、b没有保存，每次都是从Jacobian和Residual计算
 * 3、舒尔补边缘化
 *    A = Arr - Arm * Amm_inv * Amr;
 *    b = brr - Arm * Amm_inv * bmm;
 * 4、从A、b恢复Jacobian和residual，保存起来
*/
void MarginalizationInfo::marginalize()
{
    // 遍历 <将要被丢弃的变量块起始地址，0> 计算变量一维索引
    int pos = 0;
    for (auto &it : parameter_block_idx)
    {
        it.second = pos;
        pos += localSize(parameter_block_size[it.first]);
    }

    // 待丢弃的变量总数
    m = pos;

    // 遍历 <变量块起始地址, 变量块尺寸>，将保留的变量加入到 parameter_block_idx
    // 注：相当于将需要Marg的变量放到最前面
    for (const auto &it : parameter_block_size)
    {
        if (parameter_block_idx.find(it.first) == parameter_block_idx.end())
        {
            parameter_block_idx[it.first] = pos;
            pos += localSize(it.second);
        }
    }

    // 保留的变量总数
    n = pos - m;
    //ROS_INFO("marginalization, pos: %d, m: %d, n: %d, size: %d", pos, m, n, (int)parameter_block_idx.size());
    if(m == 0)
    {
        valid = false;
        printf("unstable tracking...\n");
        return;
    }

    TicToc t_summing;
    Eigen::MatrixXd A(pos, pos);
    Eigen::VectorXd b(pos);
    A.setZero();
    b.setZero();
    /*
    for (auto it : factors)
    {
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
            int idx_i = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            int size_i = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])]);
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                int idx_j = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])]);
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                if (i == j)
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else
                {
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    A.block(idx_j, idx_i, size_j, size_i) = A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    ROS_INFO("summing up costs %f ms", t_summing.toc());
    */
    //multi thread

    // 4个线程, 添加残差项，构造信息矩阵A = J^T·J， b = J^T·r
    TicToc t_thread_summing;
    pthread_t tids[NUM_THREADS];
    ThreadsStruct threadsstruct[NUM_THREADS];
    int i = 0;
    for (auto it : factors)
    {
        threadsstruct[i].sub_factors.push_back(it);
        i++;
        i = i % NUM_THREADS;
    }
    for (int i = 0; i < NUM_THREADS; i++)
    {
        TicToc zero_matrix;
        threadsstruct[i].A = Eigen::MatrixXd::Zero(pos,pos);
        threadsstruct[i].b = Eigen::VectorXd::Zero(pos);
        threadsstruct[i].parameter_block_size = parameter_block_size;
        threadsstruct[i].parameter_block_idx = parameter_block_idx;
        int ret = pthread_create( &tids[i], NULL, ThreadsConstructA ,(void*)&(threadsstruct[i]));
        if (ret != 0)
        {
            ROS_WARN("pthread_create error");
            ROS_BREAK();
        }
    }
    for( int i = NUM_THREADS - 1; i >= 0; i--)  
    {
        pthread_join( tids[i], NULL ); 
        A += threadsstruct[i].A;
        b += threadsstruct[i].b;
    }
    //ROS_DEBUG("thread summing up costs %f ms", t_thread_summing.toc());
    //ROS_INFO("A diff %f , b diff %f ", (A - tmp_A).sum(), (b - tmp_b).sum());


    /**
     * |Amm  Amr|   |delta_xm|   |bmm|
     * |        | * |        | = |   |
     * |Arm  Arr|   |delta_xr|   |brr|
     * Marg之后的A：Arr - Arm * Amm_inv * Amr
     * Marg之后的b：brr - Arm * Amm_inv * bmm
     * Marg之后：A * delta_xr = b，将delta_xm对应的信息（约束）施加到A、b上了，下次优化的时候，加上这个约束项，就是先验餐残差
    */
    Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, m, m) + A.block(0, 0, m, m).transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);

    //ROS_ASSERT_MSG(saes.eigenvalues().minCoeff() >= -1e-4, "min eigenvalue %f", saes.eigenvalues().minCoeff());

    // Amm求逆
    Eigen::MatrixXd Amm_inv = saes.eigenvectors() * 
                            Eigen::VectorXd(
                                (saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)
                            ).asDiagonal() * 
                            saes.eigenvectors().transpose();
    //printf("error1: %f\n", (Amm * Amm_inv - Eigen::MatrixXd::Identity(m, m)).sum());

    // 舒尔补边缘化
    Eigen::VectorXd bmm = b.segment(0, m);
    Eigen::MatrixXd Amr = A.block(0, m, m, n);
    Eigen::MatrixXd Arm = A.block(m, 0, n, m);
    Eigen::MatrixXd Arr = A.block(m, m, n, n);
    Eigen::VectorXd brr = b.segment(m, n);
    A = Arr - Arm * Amm_inv * Amr;
    b = brr - Arm * Amm_inv * bmm;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
    // 大于0的特征值
    Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
    // 大于0的特征值，逆
    Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

    // 求根号
    Eigen::VectorXd S_sqrt = S.cwiseSqrt();
    Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

    // ceres优化处理的是Jacobian和Residual，所以这里不保存A、b，而是恢复出J、r保存，方便下一次ceres优化
    // 从信息矩阵A恢复J，A = J^T·J
    linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    // 从b恢复残差r，b = J^T·r
    linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
    //std::cout << A << std::endl
    //          << std::endl;
    //std::cout << linearized_jacobians << std::endl;
    //printf("error2: %f %f\n", (linearized_jacobians.transpose() * linearized_jacobians - A).sum(),
    //      (linearized_jacobians.transpose() * linearized_residuals - b).sum());
}

/**
 * 保存Marg之后保留变量的一些数据
 * @param addr_shift Marg最早帧p0 [<p1,p0>,<p2,p1>,...,<pn,pn-1>], Marg倒数第二帧pn-1 [<p0,p0>, <p1,p1>,...,<pn-1,pn-1>,<pn,pn-1>]
*/
std::vector<double *> MarginalizationInfo::getParameterBlocks(std::unordered_map<long, double *> &addr_shift)
{
    // Marg之后保留变量的一些数据，这些数据只用于下一次优化，添加先验残差，不用于下一次Marg
    std::vector<double *> keep_block_addr;
    keep_block_size.clear();
    keep_block_idx.clear();
    keep_block_data.clear();

    // 遍历 <变量块起始地址，变量块一维索引> 注：需要Marg的变量索引在最前面
    for (const auto &it : parameter_block_idx)
    {
        // 保留下来的变量，m之前的都被Marg掉了
        if (it.second >= m)
        {
            // [变量块尺寸] 保留变量
            keep_block_size.push_back(parameter_block_size[it.first]);
            // [变量块索引] 保留变量 注：Marg之前的索引
            keep_block_idx.push_back(parameter_block_idx[it.first]);
            // [变量块数据] 保留变量 
            keep_block_data.push_back(parameter_block_data[it.first]);
            // [p0,p1,...,pn-1]
            keep_block_addr.push_back(addr_shift[it.first]);
        }
    }
    // 保留下来的变量块数量，13个，10个pose，1个外参，1个td，1个速度偏置
    sum_block_size = std::accumulate(std::begin(keep_block_size), std::end(keep_block_size), 0);

    return keep_block_addr;
}

/**
 * Marg先验残差
*/
MarginalizationFactor::MarginalizationFactor(MarginalizationInfo* _marginalization_info):marginalization_info(_marginalization_info)
{
    int cnt = 0;
    // [变量块尺寸] 保留变量
    for (auto it : marginalization_info->keep_block_size)
    {
        mutable_parameter_block_sizes()->push_back(it);
        cnt += it;
    }
    //printf("residual size: %d, %d\n", cnt, n);
    // 先验残差的维度
    set_num_residuals(marginalization_info->n);
};

/**
 * 迭代优化每一步调用，计算变量x在当前状态下的残差，以及残差对变量的Jacobian，用于计算delta_x，更新变量x
 * @param parameters    优化变量的值
 * @param residuals     output 残差
 * @param jacobians     output 残差对优化变量的Jacobian
*/
bool MarginalizationFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    //printf("internal addr,%d, %d\n", (int)parameter_block_sizes().size(), num_residuals());
    //for (int i = 0; i < static_cast<int>(keep_block_size.size()); i++)
    //{
    //    //printf("unsigned %x\n", reinterpret_cast<unsigned long>(parameters[i]));
    //    //printf("signed %x\n", reinterpret_cast<long>(parameters[i]));
    //printf("jacobian %x\n", reinterpret_cast<long>(jacobians));
    //printf("residual %x\n", reinterpret_cast<long>(residuals));
    //}
    // 上一次Marg操作之后保留的变量个数
    int n = marginalization_info->n;
    // 上一次Marg删除的变量个数
    int m = marginalization_info->m;
    Eigen::VectorXd dx(n);
    // 遍历 [变量块尺寸] 保留变量
    for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
    {
        // 变量块尺寸
        int size = marginalization_info->keep_block_size[i];
        // 之前保存的时候没有减去m，这里减去m，从0开始。keep_block_idx是[变量块索引] 保留变量，注：Marg之前的索引
        int idx = marginalization_info->keep_block_idx[i] - m;
        // 变量当前的值
        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
        // 上一次Marg操作之后，变量的值
        Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marginalization_info->keep_block_data[i], size);
        // dx = x - x0
        if (size != 7)
            dx.segment(idx, size) = x - x0;
        else
        {
            dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();
            dx.segment<3>(idx + 3) = 2.0 * Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0))
            {
                dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            }
        }
    }
    // 更新残差，r = r0 + J0*dx
    Eigen::Map<Eigen::VectorXd>(residuals, n) = marginalization_info->linearized_residuals + marginalization_info->linearized_jacobians * dx;
    if (jacobians)
    {

        for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
        {
            if (jacobians[i])
            {
                int size = marginalization_info->keep_block_size[i], local_size = marginalization_info->localSize(size);
                int idx = marginalization_info->keep_block_idx[i] - m;
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(jacobians[i], n, size);
                jacobian.setZero();
                // J = J0 不变
                jacobian.leftCols(local_size) = marginalization_info->linearized_jacobians.middleCols(idx, local_size);
            }
        }
    }
    return true;
}
