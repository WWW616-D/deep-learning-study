import torch
def printfa(name):
    return f'Hello {name}!'
print("hello world")
hashmap={'name':'区'}
print(hashmap.keys())
for (k,v) in hashmap.items():
    print(k,v)
def test():
    torch.tensor([1,2,3])
    print(torch.tensor([1,2,3]))
    dict={'a':1,'b':2,'c':3}
    for i,v in dict.items():
        print(i,v)
test()