# /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/18 19:19
# @Author  : Shaohd
# @FileName: request_service.py


import tornado.web
from tornado.options import define, options
from ailib import htmlize
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from ailib.exception import AIServiceException
import traceback


def format_query(query, limit=30, **kwargs):
    """
    :param query: 查询内容
    :param limit:内容限制长度
    :param kwargs:是否有其他参数
    :return: 返回处理以后的请求
    """

    if not query or query == kwargs.get('default'):
        return query
    q_unicode = query[0:limit]
    return q_unicode


define("port", default=5014, help="run on the given port", type=int)
EXECUTOR = ThreadPoolExecutor(max_workers=10)     # 多线程


class BaseService(tornado.web.RequestHandler):
    def initialize(self, runtime=None, **kwargs):
        self.debug = bool(int(self.get_argument('debug', 0)))
        self.buf = ''
        if self.debug:
            self.set_header('Content-Type', 'text/html; charset=utf-8')
            self.buf += """
            <!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN"
            "http://www.w3.org/TR/html4/strict.dtd">
            <html><head>
            <meta http-equiv="Content-Type" content="text/html;charset=utf-8">
            <meta http-equiv="Cache-Control" content="no-cache">
            </head><body>"""
            self.buf += htmlize.html_text('Enable debug mode')
        else:
            self.set_header('Content-Type', 'application/json; charset=utf-8')
            self.set_header('Cache-Control', 'no-cache')

    @tornado.web.asynchronous
    def asynchronous_get(self, default=None, **kwargs):
        """异步请求，子类在get方法中调用此方法，然后实现_get(self)方法
        """
        def callback(future):
            ex = future.exception()
            if ex is None:
                self.write_result(future.result())
            else:
                message = str(ex).replace('\n', '<br>' + ' ' * 8)
                message = message.replace('"', '&quot;')
                if isinstance(ex, AIServiceException):
                    self.write_result(default, ex.message, code=ex.code)
                else:
                    self.write_result(default, message, code=1)
            self.finish()

        return_future = EXECUTOR.submit(self.nomalization_get, **kwargs)
        return_future.add_done_callback(
            lambda future: tornado.ioloop.IOLoop.instance().add_callback(
                partial(callback, future)))

    def nomalization_get(self, **kwargs):
        if self.debug:
            try:
                return self._get(**kwargs)
            except:
                traceback.print_exc()
                raise Exception(traceback.format_exc())
        else:
            return self._get(**kwargs)


    def normal_get(self):
        self.write_result(self._get())

    def get_argument(self, name, default='', **kwargs):
        value = super(BaseService, self).get_argument(name, default)
        if not value or name == 'q':
            return value
        if name == 'rows':
            rows_limit = kwargs.pop('limit', 100)
            try:
                if int(value) > rows_limit:
                    value = '10'
            except ValueError:
                value = '10'
        if kwargs.get('limit'):
            try:
                if int(value) > kwargs['limit']:
                    value = str(kwargs['limit'])
            except ValueError:
                value = '1'
        return value

    def get_q_argument(self, default='', **kwargs):
        """
        :param default:
        :param kwargs:
        :return:
        """
        name = kwargs.pop('name', 'q')
        q = self.get_argument(name, default)
        return format_query(q, default=default, **kwargs)

    def get_int_argument(self, para_name, default=None, **kwargs):
        para_value = self.get_argument(para_name, default, **kwargs)
        if para_value == '' or para_value is None:
            if kwargs.get('nullable', False):
                if para_value is None:
                    return
            raise Exception('参数不能为空，参数：%s' % para_name)
        else:
            if isinstance(para_value, int):
                return para_value
            para_value_array = para_value.split(',')
            for value in para_value_array:
                try:
                    value = int(value)
                except Exception:
                    raise Exception('参数必须为数值型，参数：%s' % para_name)
        return para_value

    def write_result(self, result, message='successful', code=0):
        if result is None:
            result = {'totalCount': 0, 'data': []}
        if isinstance(result, dict):
            result['code'] = code
            result['message'] = message
            self.buf += htmlize.json_string(result)
        self.write(self.buf)


def run(handlers):
    tornado.options.parse_command_line()
    app = tornado.web.Application(handlers=handlers)
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()