from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

@app.route('/sum', methods=['POST'])
def sum_numbers():
    data = request.json
    numbers = data.get('numbers', [])
    result = float(np.sum(numbers))
    return jsonify({'sum': result, 'msg':'lol'})

if __name__ == '__main__':
    # print("App started")
    app.run(host='0.0.0.0', port=5000)
