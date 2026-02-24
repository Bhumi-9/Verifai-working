import requests
resp = requests.post('http://127.0.0.1:5000/api/predict', json={'review': 'I love this product, but the battery life could be better.'})
print(resp.status_code, resp.json())
