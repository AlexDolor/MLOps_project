from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

@app.route('/calculate', methods=['POST'])
def calculate_sum():
    try:
        # Получаем JSON данные из запроса
        data = request.get_json()
        
        # Проверяем, что данные получены и содержат ключ 'numbers'
        if not data or 'numbers' not in data:
            return jsonify({'error': 'Missing "numbers" field in JSON data'}), 400
        
        numbers = data['numbers']
        
        # Проверяем, что numbers является списком
        if not isinstance(numbers, list):
            return jsonify({'error': '"numbers" must be a list'}), 400
        
        # Проверяем, что все элементы списка являются числами
        if not all(isinstance(num, (int, float)) for num in numbers):
            return jsonify({'error': 'All elements in "numbers" must be numbers'}), 400
        
        # Вычисляем сумму с помощью NumPy
        numpy_array = np.array(numbers)
        total_sum = np.sum(numpy_array)
        
        # Возвращаем результат в формате JSON
        return jsonify({
            'sum': total_sum,
            'numbers_count': len(numbers),
            'status': 'success'
        })
    
    except Exception as e:
        # Обрабатываем возможные ошибки
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Эндпоинт для проверки работоспособности приложения"""
    return jsonify({'status': 'healthy', 'message': 'Service is running'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)