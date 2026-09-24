import requests, json, base64

# 1) 搜 defensive distillation 仓库
r = requests.get('https://api.github.com/search/repositories?q=defensive+distillation&sort=stars&per_page=8',
                 timeout=40, headers={'Accept': 'application/vnd.github+json'})
print('search status', r.status_code)
if r.status_code == 200:
    items = r.json()['items']
    for it in items:
        desc = (it.get('description') or '')[:60]
        print('%-45s stars=%5d %s' % (it['full_name'], it['stargazers_count'], desc))
    # 2) 抓最热门仓库的文件树, 找训练代码
    for it in items[:3]:
        repo = it['full_name']
        print('\n=== %s ===' % repo)
        t = requests.get('https://api.github.com/repos/%s/git/trees/HEAD?recursive=1' % repo,
                         timeout=40)
        if t.status_code != 200:
            print('tree fail', t.status_code); continue
        for p in t.json()['tree']:
            path = p['path']
            if path.lower().endswith(('.py', '.ipynb')) and any(k in path.lower() for k in ('train', 'distill', 'mnist', 'main')):
                print('   ', path)
else:
    print(r.text[:300])
