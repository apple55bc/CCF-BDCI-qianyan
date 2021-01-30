#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/11/30 23:00
# @Author  : QXTD-LXH
# @Desc    :
import flask
from chat import Chat
import traceback
import json
from cfg import get_logger

server = flask.Flask(__name__)
logger = get_logger()
model = Chat(logger)


@server.route('/qianyan', methods=['get', 'post'])
def qianyan():
    try:
        sample = flask.request.args.get('input')
        # sample = flask.request.get_data()
        logger.info('Data: {}'.format(sample))
        sample = json.loads(sample)
        response = model.chat(sample)
        logger.info('Response: {}'.format(response))
        return response
    except Exception as e:
        exc_info = 'Cal Exception: {}'.format(e)
        logger.info('Exception: {}'.format(traceback.format_exc()))
        logger.info('exc info: {}'.format(exc_info))
        return flask.jsonify({
            "error code": 1,
            "response": "error: {}".format(exc_info)
        })


@server.route('/test', methods=['get', 'post'])
def test():
    return 'Success ！!'


# you pig
server.run(port=2112, debug=False, host='0.0.0.0', threaded=True)
