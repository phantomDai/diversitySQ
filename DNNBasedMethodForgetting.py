# -*- encoding: utf-8 -*-
"""
@File       : DNNBasedMethodForgetting.py    
@Contact    : daihepeng@sina.cn
@Repository : https://github.com/phantomdai
@Modify Time: 2021/5/26 9:08 下午
@Author     : phantom
@Version    : 1.0
@Descriptions : 
"""
import os
import random
import logging
import time

import util.GetFeature as myFeature
import util.WriteSelectedSeeds as myWriter
import numpy as np

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


def generate_next_seed_mnist(imgs, category_seeds, candidate_test_case, category_name, calulate_method, forgetting_num):
    """
    生成下一个种子
    :param imgs: 某一个类别中的所有图像
    :param category_seeds: 某一个类别中选择的种子
    :param candidate_test_case: 候选的种子集合
    :param category_name: 类别的名字
    :param calulate_method: 计算图像差一点饿方法
    :param forgetting_num: the value of forgetting
    :return: 下一个种子
    """
    max_value = 0.
    max_candidate = ""
    for tc in candidate_test_case:
        if calulate_method == 'cosin':
            min_value = 1.
        else:
            min_value = 1000000.
        min_candidate = ''
        f1 = myFeature.get_mnist_feature(category_name, tc)
        forgetting_counter = 0
        for seed in category_seeds:
            forgetting_counter += 1
            if forgetting_counter > forgetting_num:
                break
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
                    # print("seed:" + seed)
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

def select_one_seed(imgs):
    """
    从imgs中随机选择一个种子
    :param imgs:
    :return: 选择的一个种子
    """
    return imgs[random.randint(0, len(imgs) - 1)]

def create_seed_queue_mnist(category_number, test_suite_dir, random_seed, candidate_set_size, target_dir, calculate_method):
    """
    create seed queue for mnist
    :param category_number:
    :param test_suite_dir:
    :param random_seed:
    :param candidate_set_size:
    :param target_dir:
    :param calculate_method:
    :return:
    """
    seeds_name_file = open(os.path.join(target_dir, "seeds_name"), 'a', encoding='utf-8')
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
            seeds_name_file.write(category_seeds[0] + "\n")
            first_seed_feature = myFeature.get_mnist_feature(" ", first_seed)
            temp_category = category_number[int(dir)]
            for _ in range(temp_category - 1):
                # 生成候选种子集合
                candidate_test_cases = []
                flag = True
                while flag:
                    for i in range(candidate_set_size):
                        candidate_test_cases.append(imgs[random.randint(0, len(imgs) - 1)])
                    next_seed = generate_next_seed_mnist(imgs, category_seeds, candidate_test_cases, dir, calculate_method, 10)
                    if next_seed not in category_seeds:
                        seeds_name_file.write(next_seed + "\n")
                        category_seeds.append(next_seed)
                        flag = False
            seeds.extend(category_seeds)
    seeds = [seed.replace("resize_mnist_test_category", "mnist_test_category") for seed in seeds]
    myWriter.write_selected_seeds(target_dir, seeds)
    seeds_name_file.close()

def generate_next_seed_cifar(imgs, category_seeds, candidate_test_case, category_name, calulate_method, forgetting_num):
    max_value = 0.
    max_candidate = ""
    for tc in candidate_test_case:
        if calulate_method == 'cosin':
            min_value = 1.
        else:
            min_value = 1000000.
        min_candidate = ''
        f1 = myFeature.get_cifar_feature(category_name, tc)
        forgetting_counter = 0
        for seed in category_seeds:
            forgetting_counter += 1
            if forgetting_counter > forgetting_num:
                break
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
                    dis_tc_seed = cal_distance(f1, myFeature.get_cifar_feature(category_name, seed))
                elif calulate_method == 'L2':
                    # print("seed:" + seed)
                    dis_tc_seed = cal_distance_l2(f1, myFeature.get_cifar_feature(category_name, seed))
                elif calulate_method == 'L1':
                    dis_tc_seed = cal_distance_l1(f1, myFeature.get_cifar_feature(category_name, seed))
                else:
                    dis_tc_seed = cal_distance_lmax(f1, myFeature.get_cifar_feature(category_name, seed))

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


def create_seed_queue_cifar(category_number, test_suite_dir, random_seed, candidate_set_size, target_dir, calculate_method):
    seeds_name_file = open(os.path.join(target_dir, "seeds_name"), 'a', encoding='utf-8')
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
            seeds_name_file.write(category_seeds[0] + "\n")
            first_seed_feature = myFeature.get_cifar_feature(" ", first_seed)
            temp_category = category_number[int(dir)]
            for _ in range(temp_category - 1):
                # 生成候选种子集合
                candidate_test_cases = []
                flag = True
                while flag:
                    for i in range(candidate_set_size):
                        candidate_test_cases.append(imgs[random.randint(0, len(imgs) - 1)])
                    next_seed = generate_next_seed_cifar(imgs, category_seeds, candidate_test_cases, dir,
                                                         calculate_method, 10)
                    if next_seed not in category_seeds:
                        seeds_name_file.write(next_seed + "\n")
                        category_seeds.append(next_seed)
                        flag = False
            seeds.extend(category_seeds)
    # seeds = [seed.replace("resize_mnist_test_category", "mnist_test_category") for seed in seeds]
    # myWriter.write_selected_seeds(target_dir, seeds)
    seeds_name_file.close()

if __name__ == "__main__":
    # log_path = "/Users/phantom/pythonDir/diverstySQ/results/mnist_deephunter_DNN_forgetting.txt"
    # logging.basicConfig(filename=log_path, level=logging.INFO)
    # category_number = [100, 120, 100, 100, 100, 80, 100, 100, 100, 100]
    # test_suite_dir = "/Users/phantom/dataset/mnist/resize_mnist_test_category"


    # time_1 = time.clock()
    # create_seed_queue_mnist(category_number, test_suite_dir, 0, 10, "/Users/phantom/dataset/deephunter_mnist/feature_based_10_1", 'cosin')
    # time_2 = time.clock()
    # logging.info("cosin:0:" + str((time_2 - time_1)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 1, 10, "/Users/phantom/dataset/deephunter_mnist/feature_based_10_2", 'cosin')
    # time_3 = time.clock()
    # logging.info("cosin:1:" + str((time_3 - time_2)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 2, 10, "/Users/phantom/dataset/deephunter_mnist/feature_based_10_3", 'cosin')
    # time_4 = time.clock()
    # logging.info("cosin:2:" + str((time_4 - time_3)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 3, 10, "/Users/phantom/dataset/deephunter_mnist/feature_based_10_4", 'cosin')
    # time_5 = time.clock()
    # logging.info("cosin:3:" + str((time_5 - time_4)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 4, 10, "/Users/phantom/dataset/deephunter_mnist/feature_based_10_5", 'cosin')
    # time_6 = time.clock()
    # logging.info("cosin:4:" + str((time_6 - time_5)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 5, 10, "/Users/phantom/dataset/deephunter_mnist/feature_based_10_6", 'cosin')
    # time_7 = time.clock()
    # logging.info("cosin:5:" + str((time_7 - time_6)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 6, 10, "/Users/phantom/dataset/deephunter_mnist/feature_based_10_7", 'cosin')
    # time_8 = time.clock()
    # logging.info("cosin:6:" + str((time_8 - time_7)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 7, 10, "/Users/phantom/dataset/deephunter_mnist/feature_based_10_8", 'cosin')
    # time_9 = time.clock()
    # logging.info("cosin:7:" + str((time_9 - time_8)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 8, 10, "/Users/phantom/dataset/deephunter_mnist/feature_based_10_9", 'cosin')
    # time_10 = time.clock()
    # logging.info("cosin:8:" + str((time_10 - time_9)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 9, 10, "/Users/phantom/dataset/deephunter_mnist/feature_based_10_10", 'cosin')
    # time_11 = time.clock()
    # logging.info("cosin:9:" + str((time_11 - time_10)))


    # time_12 = time.clock()
    # create_seed_queue_mnist(category_number, test_suite_dir, 0, 10, "/Users/phantom/dataset/deephunter_mnist/L2/feature_based_10_1", 'L2')
    # time_13 = time.clock()
    # # logging.info("L2:0:" + str((time_13 - time_12)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 1, 10, "/Users/phantom/dataset/deephunter_mnist/L2/feature_based_10_2", 'L2')
    # time_14 = time.clock()
    # logging.info("L2:1:" + str((time_14 - time_13)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 2, 10, "/Users/phantom/dataset/deephunter_mnist/L2/feature_based_10_3", 'L2')
    # time_15 = time.clock()
    # logging.info("L2:2:" + str((time_15 - time_14)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 3, 10, "/Users/phantom/dataset/deephunter_mnist/L2/feature_based_10_4", 'L2')
    # time_16 = time.clock()
    # logging.info("L2:3:" + str((time_16 - time_15)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 4, 10, "/Users/phantom/dataset/deephunter_mnist/L2/feature_based_10_5", 'L2')
    # time_17 = time.clock()
    # logging.info("L2:4:" + str((time_17 - time_16)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 5, 10, "/Users/phantom/dataset/deephunter_mnist/L2/feature_based_10_6", 'L2')
    # time_18 = time.clock()
    # logging.info("L2:5:" + str((time_18 - time_17)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 6, 10, "/Users/phantom/dataset/deephunter_mnist/L2/feature_based_10_7", 'L2')
    # time_19 = time.clock()
    # logging.info("L2:6:" + str((time_19 - time_18)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 7, 10, "/Users/phantom/dataset/deephunter_mnist/L2/feature_based_10_8", 'L2')
    # time_20 = time.clock()
    # logging.info("L2:7:" + str((time_20 - time_19)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 8, 10, "/Users/phantom/dataset/deephunter_mnist/L2/feature_based_10_9", 'L2')
    # time_21 = time.clock()
    # logging.info("L2:8:" + str((time_21 - time_20)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 9, 10, "/Users/phantom/dataset/deephunter_mnist/L2/feature_based_10_10", 'L2')
    # time_22 = time.clock()
    # logging.info("L2:9:" + str((time_22 - time_21)))

    # time_23 = time.clock()
    # create_seed_queue_mnist(category_number, test_suite_dir, 0, 10, "/Users/phantom/dataset/deephunter_mnist/L1/feature_based_10_1", 'L1')
    # time_24 = time.clock()
    # logging.info("L1:0:" + str((time_24 - time_23)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 1, 10, "/Users/phantom/dataset/deephunter_mnist/L1/feature_based_10_2", 'L1')
    # time_25 = time.clock()
    # logging.info("L1:1:" + str((time_25 - time_24)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 2, 10, "/Users/phantom/dataset/deephunter_mnist/L1/feature_based_10_3", 'L1')
    # time_26 = time.clock()
    # logging.info("L1:2:" + str((time_26 - time_25)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 3, 10, "/Users/phantom/dataset/deephunter_mnist/L1/feature_based_10_4", 'L1')
    # time_27 = time.clock()
    # logging.info("L1:3:" + str((time_27 - time_26)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 4, 10, "/Users/phantom/dataset/deephunter_mnist/L1/feature_based_10_5", 'L1')
    # time_28 = time.clock()
    # logging.info("L1:4:" + str((time_28 - time_27)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 5, 10, "/Users/phantom/dataset/deephunter_mnist/L1/feature_based_10_6", 'L1')
    # time_29 = time.clock()
    # logging.info("L1:5:" + str((time_29 - time_28)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 6, 10, "/Users/phantom/dataset/deephunter_mnist/L1/feature_based_10_7", 'L1')
    # time_30 = time.clock()
    # logging.info("L1:6:" + str((time_30 - time_29)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 7, 10, "/Users/phantom/dataset/deephunter_mnist/L1/feature_based_10_8", 'L1')
    # time_31 = time.clock()
    # logging.info("L1:7:" + str((time_31 - time_30)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 8, 10, "/Users/phantom/dataset/deephunter_mnist/L1/feature_based_10_9", 'L1')
    # time_32 = time.clock()
    # logging.info("L1:8:" + str((time_32 - time_31)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 9, 10, "/Users/phantom/dataset/deephunter_mnist/L1/feature_based_10_10", 'L1')
    # time_33 = time.clock()
    # logging.info("L1:9:" + str((time_33 - time_32)))

    # time_34 = time.clock()
    # create_seed_queue_mnist(category_number, test_suite_dir, 0, 10, "/Users/phantom/dataset/deephunter_mnist/Lmax/feature_based_10_1", 'Lmax')
    # time_35 = time.clock()
    # logging.info("Lmax:0:" + str((time_35 - time_34)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 1, 10, "/Users/phantom/dataset/deephunter_mnist/Lmax/feature_based_10_2", 'Lmax')
    # time_36 = time.clock()
    # logging.info("Lmax:1:" + str((time_36 - time_35)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 2, 10, "/Users/phantom/dataset/deephunter_mnist/Lmax/feature_based_10_3", 'Lmax')
    # time_37 = time.clock()
    # logging.info("Lmax:2:" + str((time_37 - time_36)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 3, 10, "/Users/phantom/dataset/deephunter_mnist/Lmax/feature_based_10_4", 'Lmax')
    # time_38 = time.clock()
    # logging.info("Lmax:3:" + str((time_38 - time_37)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 4, 10, "/Users/phantom/dataset/deephunter_mnist/Lmax/feature_based_10_5", 'Lmax')
    # time_39 = time.clock()
    # logging.info("Lmax:4:" + str((time_39 - time_38)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 5, 10, "/Users/phantom/dataset/deephunter_mnist/Lmax/feature_based_10_6", 'Lmax')
    # time_40 = time.clock()
    # logging.info("Lmax:5:" + str((time_40 - time_39)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 6, 10, "/Users/phantom/dataset/deephunter_mnist/Lmax/feature_based_10_7", 'Lmax')
    # time_41 = time.clock()
    # logging.info("Lmax:6:" + str((time_41 - time_40)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 7, 10, "/Users/phantom/dataset/deephunter_mnist/Lmax/feature_based_10_8", 'Lmax')
    # time_42 = time.clock()
    # logging.info("Lmax:7:" + str((time_42 - time_41)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 8, 10, "/Users/phantom/dataset/deephunter_mnist/Lmax/feature_based_10_9", 'Lmax')
    # time_43 = time.clock()
    # logging.info("Lmax:8:" + str((time_43 - time_42)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 9, 10, "/Users/phantom/dataset/deephunter_mnist/Lmax/feature_based_10_10", 'Lmax')
    # time_44 = time.clock()
    # logging.info("Lmax:9:" + str((time_44 - time_43)))

    log_path = "/Users/phantom/pythonDir/diverstySQ/results/cifar_deephunter_DNN_forgetting.txt"
    logging.basicConfig(filename=log_path, level=logging.INFO)
    category_number = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    test_suite_dir = "/Users/phantom/dataset/cifar10/resize_cifar_test_category"

    # time_1 = time.clock()
    # create_seed_queue_mnist(category_number, test_suite_dir, 0, 10, "/Users/phantom/dataset/deephunter_mnist/feature_based_10_1", 'cosin')
    # time_2 = time.clock()
    # logging.info("cosin:0:" + str((time_2 - time_1)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 1, 10, "/Users/phantom/dataset/deephunter_mnist/feature_based_10_2", 'cosin')
    # time_3 = time.clock()
    # logging.info("cosin:1:" + str((time_3 - time_2)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 2, 10, "/Users/phantom/dataset/deephunter_mnist/feature_based_10_3", 'cosin')
    # time_4 = time.clock()
    # logging.info("cosin:2:" + str((time_4 - time_3)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 3, 10, "/Users/phantom/dataset/deephunter_mnist/feature_based_10_4", 'cosin')
    # time_5 = time.clock()
    # logging.info("cosin:3:" + str((time_5 - time_4)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 4, 10, "/Users/phantom/dataset/deephunter_mnist/feature_based_10_5", 'cosin')
    # time_6 = time.clock()
    # logging.info("cosin:4:" + str((time_6 - time_5)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 5, 10, "/Users/phantom/dataset/deephunter_mnist/feature_based_10_6", 'cosin')


    # time_12 = time.clock()
    # create_seed_queue_mnist(category_number, test_suite_dir, 0, 10, "/Users/phantom/dataset/deephunter_mnist/L2/feature_based_10_1", 'L2')
    # time_13 = time.clock()
    # # logging.info("L2:0:" + str((time_13 - time_12)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 1, 10, "/Users/phantom/dataset/deephunter_mnist/L2/feature_based_10_2", 'L2')
    # time_14 = time.clock()
    # logging.info("L2:1:" + str((time_14 - time_13)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 2, 10, "/Users/phantom/dataset/deephunter_mnist/L2/feature_based_10_3", 'L2')
    # time_15 = time.clock()
    # logging.info("L2:2:" + str((time_15 - time_14)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 3, 10, "/Users/phantom/dataset/deephunter_mnist/L2/feature_based_10_4", 'L2')
    # time_16 = time.clock()
    # logging.info("L2:3:" + str((time_16 - time_15)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 4, 10, "/Users/phantom/dataset/deephunter_mnist/L2/feature_based_10_5", 'L2')
    # time_17 = time.clock()
    # logging.info("L2:4:" + str((time_17 - time_16)))
    # create_seed_queue_mnist(category_number, test_suite_dir, 5, 10, "/Users/phantom/dataset/deephunter_mnist/L2/feature_based_10_6", 'L2')


    time_23 = time.clock()
    create_seed_queue_cifar(category_number, test_suite_dir, 0, 10, "/Users/phantom/dataset/deephunter_cifar10/L1/feature_based_10_1", 'L1')
    time_24 = time.clock()
    logging.info("L1:0:" + str((time_24 - time_23)))
    create_seed_queue_cifar(category_number, test_suite_dir, 1, 10, "/Users/phantom/dataset/deephunter_cifar10/L1/feature_based_10_2", 'L1')
    time_25 = time.clock()
    logging.info("L1:1:" + str((time_25 - time_24)))
    create_seed_queue_cifar(category_number, test_suite_dir, 2, 10, "/Users/phantom/dataset/deephunter_cifar10/L1/feature_based_10_3", 'L1')
    time_26 = time.clock()
    logging.info("L1:2:" + str((time_26 - time_25)))
    create_seed_queue_cifar(category_number, test_suite_dir, 3, 10, "/Users/phantom/dataset/deephunter_cifar10/L1/feature_based_10_4", 'L1')
    time_27 = time.clock()
    logging.info("L1:3:" + str((time_27 - time_26)))
    create_seed_queue_cifar(category_number, test_suite_dir, 4, 10, "/Users/phantom/dataset/deephunter_cifar10/L1/feature_based_10_5", 'L1')
    time_28 = time.clock()
    logging.info("L1:4:" + str((time_28 - time_27)))


    time_34 = time.clock()
    create_seed_queue_cifar(category_number, test_suite_dir, 0, 10, "/Users/phantom/dataset/deephunter_cifar10/Lmax/feature_based_10_1", 'Lmax')
    time_35 = time.clock()
    logging.info("Lmax:0:" + str((time_35 - time_34)))
    create_seed_queue_cifar(category_number, test_suite_dir, 1, 10, "/Users/phantom/dataset/deephunter_cifar10/Lmax/feature_based_10_2", 'Lmax')
    time_36 = time.clock()
    logging.info("Lmax:1:" + str((time_36 - time_35)))
    create_seed_queue_cifar(category_number, test_suite_dir, 2, 10, "/Users/phantom/dataset/deephunter_cifar10/Lmax/feature_based_10_3", 'Lmax')
    time_37 = time.clock()
    logging.info("Lmax:2:" + str((time_37 - time_36)))
    create_seed_queue_cifar(category_number, test_suite_dir, 3, 10, "/Users/phantom/dataset/deephunter_cifar10/Lmax/feature_based_10_4", 'Lmax')
    time_38 = time.clock()
    logging.info("Lmax:3:" + str((time_38 - time_37)))
    create_seed_queue_cifar(category_number, test_suite_dir, 4, 10, "/Users/phantom/dataset/deephunter_cifar10/Lmax/feature_based_10_5", 'Lmax')
    time_39 = time.clock()
    logging.info("Lmax:4:" + str((time_39 - time_38)))
