# /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/18 19:19
# @Author  : Shaohd
# @FileName: exception.py


class AIServiceException(Exception):
    code = 1
    message = 'not correct argument!'


class ArgumentLostException(AIServiceException):
    code = 101
    base_message = '参数:%s缺失!!'

    def __init__(self, fields=[]):
        """
        构造函数.
        参数:
        fields->缺失的参数名称,可空,默认为空数组.
        """
        self.message = self.base_message % '或'.join(fields)