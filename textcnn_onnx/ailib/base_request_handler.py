# /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/18 19:19
# @Author  : Shaohd
# @FileName: base_request_handler.py


import tornado.web
from tornado.options import define, options
from ailib import htmlize


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


class BaseService(tornado.web.RequestHandler):
    def initialize(self, runtime=None, **kwargs):
        self.buf = ''
        self.set_header('Content-Type', 'application/json; charset=utf-8')
        self.set_header('Cache-Control', 'no-cache')

    def nomalization_get(self, **kwargs):
        return self._get(**kwargs)

    def normal_get(self):
        self.write_result(self._get())

    # get other param
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

    # get query text
    def get_q_argument(self, default='', **kwargs):
        name = kwargs.pop('name', 'q')
        q = self.get_argument(name, default)
        return format_query(q, default=default, **kwargs)

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