from __future__ import absolute_import, division, print_function
from flask import Flask, request, jsonify
import requests
import json
import sys

headers = {
            'Content-Type': 'application/json',
        }

data = {
            'sid': "AAATNT",
            'input_text': "Hi how are you today?"
        }
try:
    resp = requests.post(
                'http://0.0.0.0:8089/missa',
                headers=headers,
                data=json.dumps(data),
                timeout=15)
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

