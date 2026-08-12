import cv2
import pytesseract
from paddleocr import PaddleOCR
import re

# 初始化 PaddleOCR（复用已有的）
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)

# 设置 tesseract 路径
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def test_ocr_on_images(image_paths):
    """
    测试多张图片的 OCR 识别效果
    """
    results = []

    for img_path in image_paths:
        print("=" * 50)
        print(f"测试图片: {img_path}")

        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"  错误: 无法读取图片 {img_path}")
            continue

        # 转为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 二值化（白底黑字）
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # 保存预处理后的图片
        cv2.imwrite(f"processed_{img_path}", binary)

        # ========== 1. PaddleOCR 识别 ==========
        print("\n  【PaddleOCR 识别结果】")
        try:
            # PaddleOCR 需要 3 通道图片
            img_3ch = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            result = ocr.ocr(img_3ch, cls=True)

            if result and result[0]:
                for box in result[0]:
                    text = box[1][0]
                    confidence = box[1][1]
                    print(f"    文字: '{text}' (置信度: {confidence:.4f})")

                    # 提取数字
                    numbers = re.findall(r'\d+', text)
                    if numbers:
                        print(f"    提取的数字: {numbers}")
            else:
                print("    未识别到任何内容")
        except Exception as e:
            print(f"    PaddleOCR 错误: {e}")

        # ========== 2. pytesseract 识别 ==========
        print("\n  【pytesseract 识别结果】")

        # 配置1：只识别数字
        config_digits = '--psm 8 -c tessedit_char_whitelist=0123456789'
        text_digits = pytesseract.image_to_string(binary, config=config_digits)
        print(f"    只识别数字: '{text_digits.strip()}'")

        # 配置2：默认模式
        config_default = '--psm 8'
        text_default = pytesseract.image_to_string(binary, config=config_default)
        print(f"    默认模式: '{text_default.strip()}'")

        # 提取所有数字
        numbers = re.findall(r'\d+', text_default)
        if numbers:
            print(f"    提取的数字: {numbers}")

        # 存储结果
        results.append({
            'image': img_path,
            'paddleocr_texts': [box[1][0] for box in result[0]] if result and result[0] else [],
            'pytesseract_text': text_digits.strip()
        })

        print("")

    return results


# 测试的三张图片
image_files = [
    'monster_800_head.png',
    'monster_1466_head.png',
    'monster_2095_head.png'
]

# 运行测试
results = test_ocr_on_images(image_files)

# 打印汇总
print("\n" + "=" * 60)
print("汇总结果:")
for r in results:
    print(f"  {r['image']}: PaddleOCR={r['paddleocr_texts']}, pytesseract='{r['pytesseract_text']}'")