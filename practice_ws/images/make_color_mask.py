import cv2  # openCVライブラリのインポート
import numpy as np  # numpyライブラリのインポート

# inRangeを色相が0付近や180付近䛾色へ対応する形へ修正
def inRangeWrap(hsv, lower, upper):
    if lower[0] <= upper[0]:
        return cv2.inRange(hsv, lower, upper)
    else: # 180をま䛯ぐ場合
        lower1 = np.array([0, lower[1], lower[2]])
        upper1 = np.array([upper[0], upper[1], upper[2]])
        lower2 = lower
        upper2 = np.array([179, upper[1], upper[2]])
        return cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1),cv2.inRange(hsv, lower2, upper2))
 
# 画像の読み込み
img = cv2.imread("./imgs/sample1.png")
draw_img = img.copy() # 元データを書き換えないようにコピーを作成

# HSV䛻変換
hsv_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2HSV)
# BGR空間䛷䛾抽出範囲
lower = np.array([7, 50, 50]) # 色相, 彩度, 明度 下限
upper = np.array([10, 255, 255]) # 色相, 彩度, 明度 上限
# 指定範囲䛻入る画素を抽出（白が該当部分）
mask = inRangeWrap(hsv_img, lower, upper)

# マスクを適用して対象色のみ残す
result = cv2.bitwise_and(draw_img, draw_img, mask=mask)

# 画像表示
cv2.imshow("Original", draw_img)
cv2.imshow("Mask (BGR)", mask)
cv2.imshow("Result (BGR)", result)
cv2.waitKey(0)  # 何かのキーが押されるまでウィンドウを表示
