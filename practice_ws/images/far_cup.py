import cv2
import numpy as np

def inRangeWrap(hsv, lower, upper):
    if lower[0] <= upper[0]:
        return cv2.inRange(hsv, lower, upper)
    else:
        lower1 = np.array([0, lower[1], lower[2]])
        upper1 = np.array([upper[0], upper[1], upper[2]])
        lower2 = lower
        upper2 = np.array([179, upper[1], upper[2]])
        return cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1),cv2.inRange(hsv, lower2, upper2))

# 1. 画像読み込み
img = cv2.imread("ash_cup10.png")
if img is None:
    print("画像が見つかりません")
    exit()

draw_img = img.copy()
hsv_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2HSV)

# 2. 色の範囲設定（床対策）
# 床が「明るい灰色」なので、V(明度)の上限を少し厳しく(低く)して床を消します。
# もし床が白く残ってしまう場合は、Vの130を 100 くらいまで下げてください。
lower = np.array([0, 0, 0])      
upper = np.array([180, 60, 130]) # Vを150->130に下げて、明るい床を除外

mask = inRangeWrap(hsv_img, lower, upper)

# 3. モルフォロジー変換（サイズ調整）
# コップが遠くて小さいので、カーネルサイズを小さくします (15 -> 5)
kernel = np.ones((2, 2), np.uint8) 

# ノイズ除去
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# 4. 「一番大きい塊」だけを残す処理
# (背景のレンガなどが細かく反応しても、コップより小さければ消えます)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) > 0:
    # 面積が一番大きい輪郭を見つける
    max_cnt = max(contours, key=cv2.contourArea)
    
    # 新しい真っ黒なマスクを用意
    mask_final = np.zeros_like(mask)
    
    # 一番大きい輪郭だけを白く描く
    cv2.drawContours(mask_final, [max_cnt], -1, 255, -1)
    
    # マスクを更新
    mask = mask_final

# 結果確認
result = cv2.bitwise_and(draw_img, draw_img, mask=mask)

cv2.imshow("Original", draw_img)
cv2.imshow("Mask", mask)
cv2.imshow("Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()