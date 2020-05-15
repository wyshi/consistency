from __future__ import absolute_import, division, print_function
from flask import Flask, request, jsonify
import requests
import json
import sys

headers = {
            'Content-Type': 'application/json',
        }
while True:
    usr_input = input("user: ")
    data = {
                'sid': "AAATNT_copy",
                'input_text': usr_input
            }
    try:
        if usr_input == "<<quit>>":
            resp = requests.post(
                        'http://0.0.0.0:8091/user_stop',
                        headers=headers,
                        data=json.dumps(data),
                        timeout=3600)
        else:
            resp = requests.post(
                        'http://0.0.0.0:8091/persuasion_bot',
                        headers=headers,
                        data=json.dumps(data),
                        timeout=3600)
    except requests.exceptions.RequestException as e:
        # catastrophic error. bail.
        print(e)
        sys.exit(1)
    res = json.loads(resp.text)
    print(res)
    if "response" in res:
        print(res["response"])
        sent = res["response"].encode('utf-8', 'ignore').decode('utf-8')
        print(sent)

        print(res['exitbutton_appear'])
    elif "reload_success" in res:
        if res["reload_success"]:
            print("inited!")

