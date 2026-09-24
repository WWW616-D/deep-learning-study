import requests, base64

for repo, path in [('carlini/breaking_defensive_distillation', 'train_distillation.py'),
                   ('carlini/breaking_defensive_distillation', 'train_baseline.py')]:
    r = requests.get('https://api.github.com/repos/%s/contents/%s' % (repo, path), timeout=40)
    print('=== %s/%s status=%s ===' % (repo, path, r.status_code))
    if r.status_code == 200:
        src = base64.b64decode(r.json()['content']).decode('utf-8', 'ignore')
        open(r'D:\py\research_carlini_%s' % path, 'w', encoding='utf-8').write(src)
        print(src[:4000])
    else:
        print(r.text[:200])
