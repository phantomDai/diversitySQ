# -*- encoding: utf-8 -*-
"""
@File       : randomMethod.py    
@Contact    : daihepeng@sina.cn
@Repository : https://github.com/phantomdai
@Modify Time: 2021/4/12 9:06 下午
@Author     : phantom
@Version    : 1.0
@Descriptions : randomly construct a seed queue
"""

import os
import shutil
import time
import random
import util.WriteSelectedSeeds as myWriter
import util.recordInfo as myRecorder
import logging



def create_seed_queue_random(test_suite_path, num_seeds, random_seed, target_path):
    """
    randomly create a seed queue
    :param test_suite_path: the parent dir of test cases
    :param num_seeds: the number of needed seeds
    :param random_seed: the seed of random
    :param target_path: the dir that save the selected seeds
    :return: null
    """
    all_test_cases = []
    selected_seeds = []
    random.seed(random_seed)
    for root, dirs, filenames in os.walk(test_suite_path):
        for filename in filenames:
            all_test_cases.append(os.path.join(root, filename))

    for _ in range(num_seeds):
        selected_seeds.append(all_test_cases[random.randint(0, len(all_test_cases) - 1)])

    myWriter.write_selected_seeds(target_path, selected_seeds)


def create_seed_queue_mnist(test_suite_path, random_seed, target_path):
    """
    [5, 6, 5, 5, 5, 4, 5, 5, 5, 5]
    [100,120,100,100,100,80,100,100,100,100]
    :param test_suite_path:
    :param random_seed:
    :param target_path:
    :return:[100,120,100,100,100,80,100,100,100,100]
    """
    category_number = [100, 120, 100, 100, 100, 80, 100, 100, 100, 100]
    random.seed(random_seed)
    seed_queue = []
    # seeds_name_file = open(os.path.join(target_path, "seeds_name"), 'a', encoding='utf-8')

    for parent, dirs, filenames in os.walk(test_suite_path):
        for dir in dirs:
            img_dir = os.path.join(parent, dir)
            temp_list = [os.path.join(parent, dir, img) for img in os.listdir(img_dir)]
            for _ in range(category_number[int(dir)]):
                flag = True
                while flag:
                    temp_seed = temp_list[random.randint(0, len(temp_list) - 1)]
                    if temp_seed not in seed_queue:
                        seed_queue.append(temp_seed)
                        flag = False

    # for seed in seed_queue:
    #     seeds_name_file.write(seed + "\n")
    for seed in seed_queue:
        shutil.copy(seed, target_path)


def mytest():
    start_time_1 = time.clock()
    create_seed_queue_mnist("/Users/phantom/dataset/mnist/mnist_test_category", 0, \
                            "/Users/phantom/dataset/deephunter_mnist/random1")
    end_time_1 = time.clock()
    create_seed_queue_mnist("/Users/phantom/dataset/mnist/mnist_test_category", 1, \
                            "/Users/phantom/dataset/deephunter_mnist/random2")
    end_time_2 = time.clock()
    create_seed_queue_mnist("/Users/phantom/dataset/mnist/mnist_test_category", 2, \
                            "/Users/phantom/dataset/deephunter_mnist/random3")
    end_time_3 = time.clock()
    create_seed_queue_mnist("/Users/phantom/dataset/mnist/mnist_test_category", 3, \
                            "/Users/phantom/dataset/deephunter_mnist/random4")
    end_time_4 = time.clock()
    create_seed_queue_mnist("/Users/phantom/dataset/mnist/mnist_test_category", 4, \
                            "/Users/phantom/dataset/deephunter_mnist/random5")
    end_time_5 = time.clock()
    create_seed_queue_mnist("/Users/phantom/dataset/mnist/mnist_test_category", 5, \
                            "/Users/phantom/dataset/deephunter_mnist/random6")
    end_time_6 = time.clock()
    create_seed_queue_mnist("/Users/phantom/dataset/mnist/mnist_test_category", 6, \
                            "/Users/phantom/dataset/deephunter_mnist/random7")
    end_time_7 = time.clock()
    create_seed_queue_mnist("/Users/phantom/dataset/mnist/mnist_test_category", 7, \
                            "/Users/phantom/dataset/deephunter_mnist/random8")
    end_time_8 = time.clock()
    create_seed_queue_mnist("/Users/phantom/dataset/mnist/mnist_test_category", 8, \
                            "/Users/phantom/dataset/deephunter_mnist/random9")
    end_time_9 = time.clock()
    create_seed_queue_mnist("/Users/phantom/dataset/mnist/mnist_test_category", 9, \
                            "/Users/phantom/dataset/deephunter_mnist/random10")
    end_time_10 = time.clock()

    # time_info = "1: " + str(end_time_1 - start_time_1) + "\n" + "2: " + \
    #     str(end_time_2 - end_time_1) + "\n" + "3: " + \
    #     str(end_time_3 - end_time_2) + "\n" + "4: " + \
    #     str(end_time_4 - end_time_3) + "\n" + "5: " + \
    #     str(end_time_5 - end_time_4) + "\n" + "6: " + \
    #     str(end_time_6 - end_time_5) + "\n" + "7: " + \
    #     str(end_time_7 - end_time_6) + "\n" + "8: " + \
    #     str(end_time_8 - end_time_7) + "\n" + "9: " + \
    #     str(end_time_9 - end_time_8) + "\n" + "10: " + \
    #     str(end_time_10 - end_time_9)
    # myRecorder.record_time_info("/Users/phantom/pythonDir/diverstySQ/results/mnist_random.txt", time_info)


mytest()
