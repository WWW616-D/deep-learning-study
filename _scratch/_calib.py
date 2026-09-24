"""标定: 蒸馏教师/学生在 T=100 下的学习率(临时脚本, 用后即删)"""
import os, sys, importlib
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"D:\py\code")
t = importlib.import_module("try")   # try 是保留字, 不能 from try import
NormalizeNet, LeNet = t.NormalizeNet, t.LeNet
get_loaders, train_model, evaluate = t.get_loaders, t.train_model, t.evaluate

device = "cpu"
train_loader, test_loader, _ = get_loaders(128)

print("=== teacher lr=1.0 T=100 ===")
teacher = NormalizeNet(LeNet())
train_model(teacher, train_loader, test_loader, device, 8, 1.0, T=100.0, tag="teacher")
acc_t = evaluate(teacher, test_loader, device, desc="teacher final")
print(f"RESULT teacher acc={acc_t:.2f}%")

print("=== student lr=1.0 T=100 (soft labels from teacher) ===")
student = NormalizeNet(LeNet())
train_model(student, train_loader, test_loader, device, 8, 1.0, T=100.0,
            teacher=teacher, tag="student")
acc_s = evaluate(student, test_loader, device, desc="student final")
print(f"RESULT student acc={acc_s:.2f}%")
