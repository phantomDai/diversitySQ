# -*- encoding: utf-8 -*-
"""
@File       : createSQ.py    
@Contact    : daihepeng@sina.cn
@Repository : https://github.com/phantomdai
@Modify Time: 2021/6/9 2:08 下午
@Author     : phantom
@Version    : 1.0
@Descriptions : 
"""
import argparse


parse = argparse.ArgumentParser(description="Main function for creating a diversity seed queue")
parse.add_argument('seed', help='the random seed', type=int)
parse.add_argument('is_category', help='whether the test cases are stored in classification',
                   choices=[True, False], default=True)
parse.add_argument('categories_number', help='the numbers of selected test cases from each category',
                   type=list)
parse.add_argument('test_suite_dir', help='the dir that saves the test cases', type=str)
parse.add_argument('candidate_set_size', help='the size of candidate set', type=int)
parse.add_argument('is_forgetting', help='whether some selected seeds are forgotten',
                   choices=[True, False], default=False)
parse.add_argument('forgetting_number', help='the number of forgetting selected seeds', type=int)
parse.add_argument('target_dir', help='the dir that saves the selected seeds')
parse.add_argument('calculate_method', help='the method for calculating the distance between two test cases',
                   choices=['cosin', 'L1', 'L2', 'Lmax'], default='cosin')

args = parse.parse_args()

