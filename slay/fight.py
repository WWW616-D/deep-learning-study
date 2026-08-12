import subprocess
import time
import pyautogui
import mss
import cv2
import numpy as np
import keyboard
import re
from paddleocr import PaddleOCR
import pytesseract
# 初始化 PaddleOCR（全局复用，避免每次重复加载模型）
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='ch',
    use_gpu=False,
    show_log=False,
    det_db_thresh=0.1,
    det_db_box_thresh=0.1,
    det_db_unclip_ratio=2.0
    # 不指定 ocr_version，默认会用 PP-OCRv4
)
def extract_card_names(result):
    """
    从 OCR 结果中提取所有卡牌名称（排除"结束第x回合"）
    返回: list of str, 卡牌名称列表
    """
    if not result or not result[0]:
        return []
    card_names = []
    for box in result[0]:
        text = box[1][0]  # 识别的文字
        # 排除包含"结束"和"回合"的文字（如"结束第1回合"）
        if '结束' in text or '回合' in text:
            continue
        # 排除纯数字（如费用、伤害值等）
        if text.isdigit():
            continue
        # 排除血量格式（如"3/3"）
        if '/' in text:
            continue
        # 剩下的就是卡牌名称
        card_names.append(text)

    return card_names


positions=[]
def record_position():
    x, y = pyautogui.position()
    positions.append((x, y))
    print(f"[记录 #{len(positions)}] X={x}, Y={y}")
def show_all():
    print("\n" + "="*50)
    print("所有记录的坐标:")
    for i, (x, y) in enumerate(positions, 1):
        print(f"  {i}. pyautogui.click(x={x}, y={y})")
    print("="*50 + "\n")
def get_pin():
    keyboard.add_hotkey("F8",record_position)
    keyboard.add_hotkey("F9",show_all)
    keyboard.wait("F10")
def ask(pictrue):
    print("那我问你")
class slay_the_spire:
    def __init__(self):
        self.game_path=r"D:\\SteamLibrary\\steamapps\\common\\Slay the Spire 2\\SlayTheSpire2.exe"


    def StartGame(self):
        print("start game")
        subprocess.Popen(self.game_path)
        time.sleep(5)
        pyautogui.click(x=1077, y=1015)
        time.sleep(6)
        pyautogui.click(x=1077, y=1015)
        time.sleep(2)
        pyautogui.click(x=764, y=889)
        #pyautogui.click(x=2428, y=1120)
    def ContinueGame(self):
        print("continue game")
        subprocess.Popen(self.game_path)
        time.sleep(5)
        pyautogui.click(x=1077, y=1015)
        time.sleep(6)
        pyautogui.click(x=1077, y=985)
        time.sleep(8)
    def GetheroMp(self):
        img = pyautogui.screenshot(region=(115, 1176, 210, 200))
        img.save('mp.png')
        # PaddleOCR 直接接收 BGR 图像（彩色），不用转灰度
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        result = ocr.ocr(img_bgr)
        text = result[0][0][1][0]  # 例如 '91/91'
        # 分割字符串，提取两个数字
        parts = text.split('/')  # ['91', '91']
        current_mp = int(parts[0])  # 当前血量 = 91
        max_mp = int(parts[1])  # 最大血量 = 91
        print(f"当前能量: {current_mp}")
        print(f"最大能量: {max_mp}")


    def GetheroHp(self):
        img = pyautogui.screenshot(region=(417, 1039, 450, 100))
        # img.save('test.png')
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        result = ocr.ocr(img_bgr)
        text = result[0][0][1][0]  # 例如 '91/91'
        # 分割字符串，提取两个数字
        parts = text.split('/')  # ['91', '91']
        current_hp = int(parts[0])  # 当前血量 = 91
        max_hp = int(parts[1])  # 最大血量 = 91
        print(f"当前血量: {current_hp}")
        print(f"最大血量: {max_hp}")
        return current_hp, max_hp
    def GetMonster(self):
        # 1. 截取整个怪物区域
        screenshot = pyautogui.screenshot(region=(1206, 753, 1280, 420))
        img_bgr = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        # 记录原始截图区域的原点坐标（用于计算实际屏幕坐标）
        origin_x = 1206
        origin_y = 753
        # 2. 放大处理（让血量识别更准）
        scale = 2  # 放大倍数
        height, width = img_bgr.shape[:2]
        img_scaled = cv2.resize(img_bgr, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        # 3. 用 PaddleOCR 识别血量位置
        result = ocr.ocr(binary, cls=True)
        if not result or not result[0]:
            return []
        monsters = []
        for box in result[0]:
            text = box[1][0]
            if '/' not in text:
                continue
            # 解析血量
            parts = text.split('/')
            current_hp = int(parts[0])

            # 获取血量文字的中心坐标（在放大后的图片中）
            coords = box[0]
            center_x_scaled = (coords[0][0] + coords[2][0]) / 2
            center_y_scaled = (coords[0][1] + coords[2][1]) / 2

            # 转换为原始截图中的坐标（未放大）
            center_x_original = center_x_scaled / scale
            center_y_original = center_y_scaled / scale

            # 转换为实际屏幕坐标
            actual_x = origin_x + center_x_original
            actual_y = origin_y + center_y_original

            # 怪物实际坐标（血量真实坐标往上 80 像素）
            monster_x = actual_x
            monster_y = actual_y - 80

            # 4. 往头顶区域裁剪（在放大后的图片上裁剪用于识别）
            head_top = int(center_y_scaled - 450)
            head_bottom = int(center_y_scaled - 250)
            head_left = int(center_x_scaled - 100)
            head_right = int(center_x_scaled + 100)

            # 确保不越界
            head_top = max(0, head_top)
            head_bottom = min(binary.shape[0], head_bottom)
            head_left = max(0, head_left)
            head_right = min(binary.shape[1], head_right)

            # 裁剪头顶区域（二值化后的，用于识别）
            head_roi = binary[head_top:head_bottom, head_left:head_right]

            # 保存调试图片
            cv2.imwrite(f'monster_{center_x_scaled:.0f}_head.png', head_roi)

            # 5. 用 PaddleOCR 识别头顶区域的数字
            head_roi_bgr = cv2.cvtColor(head_roi, cv2.COLOR_GRAY2BGR)
            damage_result = ocr.ocr(head_roi_bgr, cls=True)

            # 提取伤害数字
            damage = 0
            recognized_text = ""

            if damage_result and damage_result[0]:
                for damage_box in damage_result[0]:
                    damage_text = damage_box[1][0]
                    recognized_text += damage_text + " "
                    # 提取纯数字
                    damage_numbers = re.findall(r'\d+', damage_text)
                    if damage_numbers:
                        damage = int(damage_numbers[0])
                        break

            print(f"血量: {current_hp}, 头顶识别: '{recognized_text.strip()}' -> 伤害: {damage}")
            print(f"怪物实际坐标: ({monster_x:.0f}, {monster_y:.0f})")

            # 6. 当未识别出数字或伤害小于等于2时，调用 ask 函数
            if damage == 0 or damage <= 2:
                # 保存原始截图（未处理的头顶区域）
                original_head_roi = img_scaled[head_top:head_bottom, head_left:head_right]
                cv2.imwrite(f'ask_monster_{center_x_scaled:.0f}_head.png', original_head_roi)
                # 调用 ask 函数，传入原始截图路径和怪物坐标
                answer = ask(f'ask_monster_{center_x_scaled:.0f}_head.png')
            monsters.append({
                'hp': current_hp,
                'damage': damage,
                'position': (monster_x, monster_y)  # 记录怪物实际坐标
            })
        # 打印结果
        for i, m in enumerate(monsters):
            print(f"怪物{i + 1}: 血量 {m['hp']}, 伤害 {m['damage']}, 坐标 {m['position']}")
        return monsters
    def GetCard(self):
        img = pyautogui.screenshot(region=(100, 1183, 2500, 320))
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        result = ocr.ocr(img_bgr, cls=True)
        card=extract_card_names(result)
        print(card)
    def GetHero(self):
        self.GetheroHp()
        self.GetheroMp()
    def Fight(self):
        print("Fight")
        while True:
            # 检测是否按下停止键
            if keyboard.is_pressed('F12'):
                print("用户停止战斗")
                break
            self.GetHero()
            self.GetMonster()
            self.GetCard()
def main():
    print("start game")
    game=slay_the_spire()
    #game.StartGame()
    game.ContinueGame()
    game.Fight()
    #get_pin()


main()