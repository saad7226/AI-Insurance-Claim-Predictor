from flask import request, jsonify

def require_auth(f):
    def wrapper(*args, **kwargs):
        api_key = request.headers.get('Authorization')
        if api_key != 'saad7223':  
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper