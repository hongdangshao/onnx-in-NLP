# /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/18 19:19
# @Author  : Shaohd
# @FileName: htmlize.py

import json
import ujson
import pprint

FONT_FORMAT = """<font STYLE="font-size: small; font-family:'sans-serif';">%s</font>"""
BOLD_FONT_FORMAT = """<b><font STYLE="font-size: small; font-family:'sans-serif';">%s</font></b>"""
WARNING_FONT_FORMAT = """<b><font color="red" STYLE="font-size: small; font-family:'sans-serif'";>%s</font></b>"""
PRE_FORMAT = """<pre STYLE="font-size: small; font-family:'sans-serif'; white-space: pre-wrap; wrap=on">%s</pre>"""
LABEL_FORMAT = """<label STYLE="font-size: small; font-family:'sans-serif'; white-space: pre-wrap; wrap=on"><b>%s</b></label>"""
HTTP_400_ERROR = """<html><body>The request could not be processed by the server, make sure your parameters are valid and try again.</body></html>"""

JSON_HTTP_HEADER = """Content-Type: application/json; charset=utf-8
Cache-Control: no-cache"""

HTML_HTTP_HEADER = """Content-Type: text/html; charset=utf-8"""


def html_pre(obj, label=''):
    html = ''
    if label:
        html += FONT_FORMAT % (label)
    pp = pprint.PrettyPrinter(indent=4)
    html += PRE_FORMAT % (pp.pformat(obj))
    return html


def html_cache_status_table(dic, label=''):
    html = """
<style type="text/css">
#customers
{
    width:100%;
    border-collapse:collapse;
}
#customers td, #customers th
{
    border:1px solid #98bf21;
    padding:3px 7px 2px 7px;
}
#customers th
{
    text-align:left;
    padding-top:5px;
    padding-bottom:4px;
    background-color:#A7C942;
    color:#ffffff;
}
#customers tr.alt td
{
    color:#000000;
    background-color:#EAF2D3;
}
</style>
"""

    if label:
        html += FONT_FORMAT % (label)

    table_header = """
    <table id="customers">
    <tr>
    <th>Doctor Keys</th>
    <th>Cached</th>
    <th>Inprogress</th>
    <th>Missed</th>
    <th>No Ticket</th>
    </tr>
"""
    html += table_header

    cached = dic['cached']
    missed = dic['missed']
    inprogress = dic['inprogress']
    noticket = dic['noticket']

    td = lambda s: '<td>' + s + '</td>'
    tr = lambda s: '<tr>' + s + '</tr>'

    keys = cached + missed + inprogress + noticket
    keys.sort()
    for key in keys:
        # all keys
        s = td(key)
        # cached
        if key in cached:
            s += td('Y')
        else:
            s += td('N')
        # inprogess
        if key in inprogress:
            s += td('Y')
        else:
            s += td('N')
        # missed
        if key in missed:
            s += td('Y')
        else:
            s += td('N')
        # no ticket
        if key in noticket:
            s += td('Y')
        else:
            s += td('N')
        html += tr(s)
    html += '</table>'

    return html


def html_text(s):
    html = LABEL_FORMAT % (s)
    html += '<br>'
    return html


def html_string(name, s):
    html = FONT_FORMAT % (name)
    html += FONT_FORMAT % (str(s))
    html += '<br>'
    return html


def html_indented_json(s, label='', debug=False):
    html = ''
    if label:
        html += BOLD_FONT_FORMAT % (label)
    if debug:
        json_html = json.dumps(s, sort_keys=True, indent=4, ensure_ascii=False)
    else:
        json_html = ujson.dumps(s, ensure_ascii=False)
    html += PRE_FORMAT % (json_html)
    return html


def html_json(s):
    return ujson.dumps(s, ensure_ascii=False)


def json_string(dic):
    return ujson.dumps(dic, ensure_ascii=False)