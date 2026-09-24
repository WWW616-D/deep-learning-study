import requests, base64

r = requests.get('https://api.github.com/repos/carlini/breaking_defensive_distillation/contents/model.py', timeout=40)
print('status', r.status_code)
if r.status_code == 200:
    src = base64.b64decode(r.json()['content']).decode('utf-8', 'ignore')
    open(r'D:\py\research_carlini_model.py', 'w', encoding='utf-8').write(src)
    print(src[:3000])
else:
    print(r.text[:300])
