# -*- encoding: utf-8 -*-
"""
@File       : DNNBasedMethod.py    
@Contact    : daihepeng@sina.cn
@Repository : https://github.com/phantomdai
@Modify Time: 2021/4/8 8:31 下午
@Author     : phantom
@Version    : 1.0
@Descriptions : create seed queue based on VGG19
"""
import time

import numpy as np
import random
import os
import util.WriteSelectedSeeds as myWriter
import util.recordInfo as myRecorder
import util.GetFeature as myFeature
import util.GetFeatures as imgFeature


first_seed = ""
first_seed_feature = [[]]


def cal_distance_l2(f1, f2):
    """
    calculate the L2 distance of f1 and f2
    :param f1:
    :param f2:
    :return:
    """
    return np.linalg.norm(f1 - f2, ord=2)


def cal_distance_l1(f1, f2):
    """
    calculate the L1 distance of f1 and f2
    :param f1:
    :param f2:
    :return:
    """
    return np.linalg.norm(f1 - f2, ord=1)


def cal_distance_lmax(f1, f2):
    """
    calculate the L max distance of f1 and f2
    :param f1:
    :param f2:
    :return:
    """
    return np.linalg.norm(f1 - f2, ord=np.inf)


def cal_distance(f1, f2):
    """
    calculate the distance of f1 and f2
    :param f1: the vector f1
    :param f2: the vector f2
    :return: the value of cosine
    """

    cos1 = np.sum(f1 * f2)
    cos21 = np.sqrt(np.sum(f1 * f1))
    cos22 = np.sqrt(np.sum(f2 * f2))
    return cos1 / float(cos22 * cos21)


def select_one_seed(imgs):
    """
    从imgs中随机选择一个种子
    :param imgs:
    :return: 选择的一个种子
    """
    return imgs[random.randint(0, len(imgs) - 1)]


def generate_next_seed_imagenet(seeds, candidate_test_cases, random_seed, calulate_method):
    """
    从候选测试用例集中选择一个测试用例进行执行
    :param seeds: 种子集合
    :param candidate_test_cases: 候选种子集合
    :param calulate_method: 计算测试用例距离的方法
    :param random_seed: 随机数的种子
    :return:
    """
    max_value = 0.
    max_candidate = ""
    for tc in candidate_test_cases:
        if calulate_method == 'cosin':
            min_value = 1.
        else:
            min_value = 1000000.
        min_candidate = ''
        f1 = myFeature.get_feature_imagenet(len(candidate_test_cases), random_seed, tc)

        print("f1:" + tc + str(f1))
        for seed in seeds:
            if seed == first_seed:
                if calulate_method == 'cosin':
                    dis_tc_seed = cal_distance(f1, first_seed_feature)
                elif calulate_method == 'L2':
                    dis_tc_seed = cal_distance_l2(f1, first_seed_feature)
                elif calulate_method == 'L1':
                    dis_tc_seed = cal_distance_l1(f1, first_seed_feature)
                else:
                    dis_tc_seed = cal_distance_lmax(f1, first_seed_feature)
            else:
                if calulate_method == 'cosin':
                    dis_tc_seed = cal_distance(f1, myFeature.get_feature_imagenet(len(candidate_test_cases), random_seed, seed))
                elif calulate_method == 'L2':
                    dis_tc_seed = cal_distance_l2(f1, myFeature.get_feature_imagenet(len(candidate_test_cases), random_seed, seed))
                elif calulate_method == 'L1':
                    dis_tc_seed = cal_distance_l1(f1, myFeature.get_feature_imagenet(len(candidate_test_cases), random_seed, seed))
                else:
                    dis_tc_seed = cal_distance_lmax(f1, myFeature.get_feature_imagenet(len(candidate_test_cases), random_seed, seed))
            if dis_tc_seed < min_value:
                min_value = dis_tc_seed
                min_candidate = tc
            else:
                pass

        if min_value > max_value:
            max_value = min_value
            max_candidate = min_candidate
        else:
            pass

    print("选择的种子:" + max_candidate)
    return max_candidate


def create_seed_queue_imagenet(random_seed, candidate_set_size, seeds_size, target_dir, calculate_method):
    """
    为ImageNet生成种子队列
    :param random_seed: 随机数种子
    :param candidate_set_size: 候选种子集合的大小
    :param seeds_size: 种子队列的大小
    :param target_dir: 将产生的种子放入指定的目录
    :param calculate_method: 计算测试用例距离的方式
    :return:
    """
    test_suite_dir = "/Users/phantom/dataset/ImageNet/imageNet_test"
    # 获取所有的测试用例
    all_test_cases = []
    for parent, dirs, filenames in os.walk(test_suite_dir):
        for filename in filenames:
            all_test_cases.append(os.path.join(parent, filename))

    # 随机选择一个测试用例作为第一个种子
    seeds = []
    seeds.append(all_test_cases[random.randint(0, len(all_test_cases) - 1)])
    global first_seed, first_seed_feature
    first_seed = seeds[0]
    first_seed_feature = imgFeature.get_feature(first_seed)


    random.seed(random_seed)
    for _ in range(seeds_size - 1):
        # 生成候选种子集合
        candidate_test_cases = []
        for i in range(candidate_set_size):
            candidate_test_cases.append(all_test_cases[random.randint(0, len(all_test_cases) - 1)])

        # for tc in candidate_test_cases:
        #     print("我要打印tc:" + tc)
        seeds.append(generate_next_seed_imagenet(seeds, candidate_test_cases, random_seed, calculate_method))

    myWriter.write_selected_seeds(target_dir, seeds)


def generate_next_seed_mnist(imgs, category_seeds, candidate_test_case, category_name, calulate_method):
    """
    生成下一个种子
    :param imgs: 某一个类别中的所有图像
    :param category_seeds: 某一个类别中选择的种子
    :param candidate_test_case: 候选的种子集合
    :param category_name: 类别的名字
    :param calulate_method: 计算图像差一点饿方法
    :return: 下一个种子
    """
    for seed in category_seeds:
        print(seed)
    max_value = 0.
    max_candidate = ""
    for tc in candidate_test_case:
        if calulate_method == 'cosin':
            min_value = 1.
        else:
            min_value = 1000000.
        min_candidate = ''
        f1 = myFeature.get_mnist_feature(category_name, tc)
        print("f1:" + tc + str(f1))
        for seed in category_seeds:
            if seed == first_seed:
                if calulate_method == 'cosin':
                    dis_tc_seed = cal_distance(f1, first_seed_feature)
                elif calulate_method == 'L2':
                    dis_tc_seed = cal_distance_l2(f1, first_seed_feature)
                elif calulate_method == 'L1':
                    dis_tc_seed = cal_distance_l1(f1, first_seed_feature)
                else:
                    dis_tc_seed = cal_distance_lmax(f1, first_seed_feature)
            else:
                if calulate_method == 'cosin':
                    dis_tc_seed = cal_distance(f1, myFeature.get_mnist_feature(category_name, seed))
                elif calulate_method == 'L2':
                    print("seed:" + seed)
                    dis_tc_seed = cal_distance_l2(f1, myFeature.get_mnist_feature(category_name, seed))
                elif calulate_method == 'L1':
                    dis_tc_seed = cal_distance_l1(f1, myFeature.get_mnist_feature(category_name, seed))
                else:
                    dis_tc_seed = cal_distance_lmax(f1, myFeature.get_mnist_feature(category_name, seed))

            if dis_tc_seed < min_value:
                min_value = dis_tc_seed
                min_candidate = tc
            else:
                pass

        if min_value > max_value:
            max_value = min_value
            max_candidate = min_candidate
        else:
            pass
    print("选择的种子:" + max_candidate)
    return max_candidate

def generate_next_seed_mnist_with_forgetting(imgs, category_seeds, candidate_test_case, category_name, calulate_method):
    """
        生成下一个种子
        :param imgs: 某一个类别中的所有图像
        :param category_seeds: 某一个类别中选择的种子
        :param candidate_test_case: 候选的种子集合
        :param category_name: 类别的名字
        :param calulate_method: 计算图像差一点饿方法
        :return: 下一个种子
        """
    for seed in category_seeds:
        print(seed)
    max_value = 0.
    max_candidate = ""
    for tc in candidate_test_case:
        if calulate_method == 'cosin':
            min_value = 1.
        else:
            min_value = 1000000.
        min_candidate = ''
        f1 = myFeature.get_mnist_feature(category_name, tc)
        print("f1:" + tc + str(f1))
        for seed in category_seeds:
            if seed == first_seed:
                if calulate_method == 'cosin':
                    dis_tc_seed = cal_distance(f1, first_seed_feature)
                elif calulate_method == 'L2':
                    dis_tc_seed = cal_distance_l2(f1, first_seed_feature)
                elif calulate_method == 'L1':
                    dis_tc_seed = cal_distance_l1(f1, first_seed_feature)
                else:
                    dis_tc_seed = cal_distance_lmax(f1, first_seed_feature)
            else:
                if calulate_method == 'cosin':
                    dis_tc_seed = cal_distance(f1, myFeature.get_mnist_feature(category_name, seed))
                elif calulate_method == 'L2':
                    print("seed:" + seed)
                    dis_tc_seed = cal_distance_l2(f1, myFeature.get_mnist_feature(category_name, seed))
                elif calulate_method == 'L1':
                    dis_tc_seed = cal_distance_l1(f1, myFeature.get_mnist_feature(category_name, seed))
                else:
                    dis_tc_seed = cal_distance_lmax(f1, myFeature.get_mnist_feature(category_name, seed))

            if dis_tc_seed < min_value:
                min_value = dis_tc_seed
                min_candidate = tc
            else:
                pass

        if min_value > max_value:
            max_value = min_value
            max_candidate = min_candidate
        else:
            pass
    print("选择的种子:" + max_candidate)
    return max_candidate

def create_seed_queue_mnist(random_seed, candidate_set_size, target_dir, calculate_method):
    """
    为mnist数据集生成基于深度特征的种子队列
    :param random_seed: 随机数的种子
    :param candidate_set_size: 候选测试用例集的大小
    :param target_dir: 将生成的测试用例移动到指定的目录
    :param calculate_method: 计算图像差异的方法
    :return:
    """
    category_number = [100, 120, 100, 100, 100, 80, 100, 100, 100, 100]
    test_suite_dir = "/Users/phantom/dataset/mnist/resize_mnist_test_category"
    # random.seed(random_seed)
    # 存放选择的种子
    seeds = []
    for parent, dirs, filenames in os.walk(test_suite_dir):
        for dir in dirs:
            random.seed(random_seed)
            # 某一个类别中选择的种子
            category_seeds = []
            category_dir = os.path.join(parent, dir)
            imgs = [os.path.join(parent, dir, img_name) for img_name in os.listdir(category_dir)]
            # 随机选一个种子作为初始种子
            category_seeds.append(select_one_seed(imgs))
            global first_seed, first_seed_feature
            first_seed = category_seeds[0]
            first_seed_feature = myFeature.get_mnist_feature(" ", first_seed)
            temp_category = category_number[int(dir)]
            for _ in range(temp_category - 1):
                # 生成候选种子集合
                candidate_test_cases = []
                for i in range(candidate_set_size):
                    candidate_test_cases.append(imgs[random.randint(0, len(imgs) - 1)])

                # for tc in candidate_test_cases:
                #     print(tc)

                category_seeds.append(generate_next_seed_mnist(imgs, category_seeds, candidate_test_cases, dir, calculate_method))
            seeds.extend(category_seeds)
            print("seeds:" + " ".join(seeds))
    seeds = [seed.replace("resize_mnist_test_category", "mnist_test_category") for seed in seeds]
    myWriter.write_selected_seeds(target_dir, seeds)


if __name__ == '__main__':
    # create_seed_queue_imagenet(1, 10, 20, "/Users/phantom/dataset/deephunter_mnist/feature_based_10_1", 'cosin')
    # create_seed_queue_imagenet(2, 10, 20, "/Users/phantom/dataset/deephunter_mnist/feature_based_10_2", 'cosin')
    # create_seed_queue_imagenet(3, 10, 20, "/Users/phantom/dataset/deephunter_mnist/feature_based_10_3", 'cosin')
    # create_seed_queue_imagenet(4, 10, 20, "/Users/phantom/dataset/deephunter_mnist/feature_based_10_4", 'cosin')
    # create_seed_queue_imagenet(5, 10, 20, "/Users/phantom/dataset/deephunter_mnist/feature_based_10_5", 'cosin')

    # create_seed_queue_imagenet(1, 10, 20, "/Users/phantom/dataset/deephunter_mnist/L2/feature_based_10_1", 'L2')
    # create_seed_queue_imagenet(2, 10, 20, "/Users/phantom/dataset/deephunter_mnist/L2/feature_based_10_2", 'L2')
    # create_seed_queue_imagenet(3, 10, 20, "/Users/phantom/dataset/deephunter_mnist/L2/feature_based_10_3", 'L2')
    # create_seed_queue_imagenet(4, 10, 20, "/Users/phantom/dataset/deephunter_mnist/L2/feature_based_10_4", 'L2')
    # create_seed_queue_imagenet(5, 10, 20, "/Users/phantom/dataset/deephunter_mnist/L2/feature_based_10_5", 'L2')

    # create_seed_queue_imagenet(1, 10, 20, "/Users/phantom/dataset/deephunter_mnist/Lmax/feature_based_10_1", 'Lmax')
    # create_seed_queue_imagenet(2, 10, 20, "/Users/phantom/dataset/deephunter_mnist/Lmax/feature_based_10_2", 'Lmax')
    # create_seed_queue_imagenet(3, 10, 20, "/Users/phantom/dataset/deephunter_mnist/Lmax/feature_based_10_3", 'Lmax')
    # create_seed_queue_imagenet(4, 10, 20, "/Users/phantom/dataset/deephunter_mnist/Lmax/feature_based_10_4", 'Lmax')
    # create_seed_queue_imagenet(5, 10, 20, "/Users/phantom/dataset/deephunter_mnist/Lmax/feature_based_10_5", 'Lmax')

    create_seed_queue_mnist(0, 10, "/Users/phantom/dataset/deephunter_mnist/feature_based_10_1", 'cosin')
    # create_seed_queue_mnist(1, 10, "/Users/phantom/dataset/deephunter_mnist/feature_based_10_2", 'cosin')
    # create_seed_queue_mnist(2, 10, "/Users/phantom/dataset/deephunter_mnist/feature_based_10_3", 'cosin')
    # create_seed_queue_mnist(3, 10, "/Users/phantom/dataset/deephunter_mnist/feature_based_10_4", 'cosin')
    # create_seed_queue_mnist(4, 10, "/Users/phantom/dataset/deephunter_mnist/feature_based_10_5", 'cosin')
    # create_seed_queue_mnist(5, 10, "/Users/phantom/dataset/deephunter_mnist/feature_based_10_6", 'cosin')
    # create_seed_queue_mnist(6, 10, "/Users/phantom/dataset/deephunter_mnist/feature_based_10_7", 'cosin')
    # create_seed_queue_mnist(7, 10, "/Users/phantom/dataset/deephunter_mnist/feature_based_10_8", 'cosin')
    # create_seed_queue_mnist(8, 10, "/Users/phantom/dataset/deephunter_mnist/feature_based_10_9", 'cosin')
    # create_seed_queue_mnist(9, 10, "/Users/phantom/dataset/deephunter_mnist/feature_based_10_10", 'cosin')


    # create_seed_queue_mnist(0, 10, "/Users/phantom/dataset/deephunter_mnist/L2/feature_based_10_1", 'L2')
    # create_seed_queue_mnist(1, 10, "/Users/phantom/dataset/deephunter_mnist/L2/feature_based_10_2", 'L2')
    # create_seed_queue_mnist(2, 10, "/Users/phantom/dataset/deephunter_mnist/L2/feature_based_10_3", 'L2')
    # create_seed_queue_mnist(3, 10, "/Users/phantom/dataset/deephunter_mnist/L2/feature_based_10_4", 'L2')
    # create_seed_queue_mnist(4, 10, "/Users/phantom/dataset/deephunter_mnist/L2/feature_based_10_5", 'L2')
    # create_seed_queue_mnist(5, 10, "/Users/phantom/dataset/deephunter_mnist/L2/feature_based_10_6", 'L2')
    # create_seed_queue_mnist(6, 10, "/Users/phantom/dataset/deephunter_mnist/L2/feature_based_10_7", 'L2')
    # create_seed_queue_mnist(7, 10, "/Users/phantom/dataset/deephunter_mnist/L2/feature_based_10_8", 'L2')
    # create_seed_queue_mnist(8, 10, "/Users/phantom/dataset/deephunter_mnist/L2/feature_based_10_9", 'L2')
    # create_seed_queue_mnist(9, 10, "/Users/phantom/dataset/deephunter_mnist/L2/feature_based_10_10", 'L2')

    # create_seed_queue_mnist(0, 10, "/Users/phantom/dataset/deephunter_mnist/L1/feature_based_10_1", 'L1')
    # create_seed_queue_mnist(1, 10, "/Users/phantom/dataset/deephunter_mnist/L1/feature_based_10_2", 'L1')
    # create_seed_queue_mnist(2, 10, "/Users/phantom/dataset/deephunter_mnist/L1/feature_based_10_3", 'L1')
    # create_seed_queue_mnist(3, 10, "/Users/phantom/dataset/deephunter_mnist/L1/feature_based_10_4", 'L1')
    # create_seed_queue_mnist(4, 10, "/Users/phantom/dataset/deephunter_mnist/L1/feature_based_10_5", 'L1')
    # create_seed_queue_mnist(5, 10, "/Users/phantom/dataset/deephunter_mnist/L1/feature_based_10_6", 'L1')
    # create_seed_queue_mnist(6, 10, "/Users/phantom/dataset/deephunter_mnist/L1/feature_based_10_7", 'L1')
    # create_seed_queue_mnist(7, 10, "/Users/phantom/dataset/deephunter_mnist/L1/feature_based_10_8", 'L1')
    # create_seed_queue_mnist(8, 10, "/Users/phantom/dataset/deephunter_mnist/L1/feature_based_10_9", 'L1')
    # create_seed_queue_mnist(9, 10, "/Users/phantom/dataset/deephunter_mnist/L1/feature_based_10_10", 'L1')

    # create_seed_queue_mnist(0, 10, "/Users/phantom/dataset/deephunter_mnist/Lmax/feature_based_10_1", 'Lmax')
    # create_seed_queue_mnist(1, 10, "/Users/phantom/dataset/deephunter_mnist/Lmax/feature_based_10_2", 'Lmax')
    # create_seed_queue_mnist(2, 10, "/Users/phantom/dataset/deephunter_mnist/Lmax/feature_based_10_3", 'Lmax')
    # create_seed_queue_mnist(3, 10, "/Users/phantom/dataset/deephunter_mnist/Lmax/feature_based_10_4", 'Lmax')
    # create_seed_queue_mnist(4, 10, "/Users/phantom/dataset/deephunter_mnist/Lmax/feature_based_10_5", 'Lmax')
    # create_seed_queue_mnist(5, 10, "/Users/phantom/dataset/deephunter_mnist/Lmax/feature_based_10_6", 'Lmax')
    # create_seed_queue_mnist(6, 10, "/Users/phantom/dataset/deephunter_mnist/Lmax/feature_based_10_7", 'Lmax')
    # create_seed_queue_mnist(7, 10, "/Users/phantom/dataset/deephunter_mnist/Lmax/feature_based_10_8", 'Lmax')
    # create_seed_queue_mnist(8, 10, "/Users/phantom/dataset/deephunter_mnist/Lmax/feature_based_10_9", 'Lmax')
    # create_seed_queue_mnist(9, 10, "/Users/phantom/dataset/deephunter_mnist/Lmax/feature_based_10_10", 'Lmax')
