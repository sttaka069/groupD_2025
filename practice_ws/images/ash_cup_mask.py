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

# 画像読み込み
img = cv2.imread("ash_cup7.png")
if img is None: exit()
draw_img = img.copy()
hsv_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2HSV)

# 色の範囲
lower = np.array([0, 0, 0])      
upper = np.array([180, 50, 150]) 
mask = inRangeWrap(hsv_img, lower, upper)

# モルフォロジー変換（穴埋め）
# ※輪郭を見つける前に、コップを「ひとかたまり」にする
kernel = np.ones((2, 2), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# 輪郭（白い塊）を検出する
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 画像の幅を取得
img_h, img_w = mask.shape

# 見つかった塊を一つずつチェックするループ
for cnt in contours:
    # 塊の情報を取得（x, y: 左上の座標, w: 幅, h: 高さ）
    x, y, w, h = cv2.boundingRect(cnt)
    
    # 塊の中心(X座標)を計算
    center_x = x + (w / 2)
    
    # 判定：中心が「画面の真ん中より左」にあるか？
    if center_x < (img_w / 2):
        # 左にある塊（箱だとおもうから）なら、その部分を黒で塗りつぶして消す
        # drawContours(画像, [輪郭データ], インデックス, 色, 太さ(-1は塗りつぶし))
        cv2.drawContours(mask, [cnt], -1, 0, -1)

result = cv2.bitwise_and(draw_img, draw_img, mask=mask)

cv2.imshow("Original", draw_img)
cv2.imshow("Smart Mask", mask)
cv2.imshow("Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()