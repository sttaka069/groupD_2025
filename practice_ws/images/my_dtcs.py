import cv2  # openCVライブラリのインポート
import numpy as np  # numpyライブラリのインポート
from cv2 import aruco, imread, imwrite

##　↓↓↓↓↓↓↓inRangeWrap, calc_centroidは変更しないでください↓↓↓↓↓↓
# inRangeを色相が0付近や180付近の色へ対応する形へ修正
def inRangeWrap(hsv, lower, upper):
    if lower[0] <= upper[0]:
        return cv2.inRange(hsv, lower, upper)
    else:
        # 180をまたぐ場合
        lower1 = np.array([0, lower[1], lower[2]])
        upper1 = np.array([upper[0], upper[1], upper[2]])
        lower2 = lower
        upper2 = np.array([179, upper[1], upper[2]])
        return cv2.bitwise_or(
            cv2.inRange(hsv, lower1, upper1),
            cv2.inRange(hsv, lower2, upper2)
        )
    
def calc_centroid(mask):
    M = cv2.moments(mask)
    if M["m00"] != 0:
        # 重心座標を計算SS
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        s = np.count_nonzero(mask)/(mask.shape[0]*mask.shape[1]) #画像に占めるマスクの割合
        return cx, cy, s
    else:
        return None   
##　↑↑↑↑↑↑↑inRangeWrap, calc_centroidは変更しないでください↑↑↑↑↑↑↑

def orange_ball(img):
    draw_img = img.copy() # 元データを書き換えないようにコピーを作成
    # HSVに変換
    hsv_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 202, 70]) 
    upper = np.array([10, 253, 255])

    mask = cv2.inRange(hsv_img, lower, upper)
    
    # ノイズ除去（重要）
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 小さな点を消す
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # ボールの中の穴を埋める

    try:
        x, y, s = calc_centroid(mask)
        print(f"{s=}")
        return x, y
    except TypeError:
        return None

def d_coke(img):
    # 画像の読み込み
    draw_img = img.copy() # 元データを書き換えないようにコピーを作成
    # HSVに変換（色指定はRGBよりHSVの方が扱いやすい）
    hsv_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2HSV)

    # BGR空間での抽出範囲
    ## コーラ缶
    lower_target = np.array([170, 189, 62])
    upper_target = np.array([179, 255, 220])

    # 指定範囲に入る画素を抽出（白が該当部分）
    mask = inRangeWrap(hsv_img, lower, upper)
    
    # ノイズ除去（重要）
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 小さな点を消す
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # ボールの中の穴を埋める
    
    try:
        x, y, s = calc_centroid(mask)
        print(f"{s=}")
        return x, y
    except TypeError:
        return None

def d_circle(img):
    # 画像読み込み
    draw_img = img.copy()

    # 前処理（グレースケール＋ぼかし）
    gray = cv2.cvtColor(draw_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)  # ノイズ低減

    # 円検出（HoughCircles）
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=30,
        param1=100, param2=50,   
        minRadius=10, maxRadius=40  
    )
    # マスク作成（検出円を塗りつぶし）
    mask = np.zeros(gray.shape, dtype=np.uint8)
    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        for x, y, r in circles:
            cv2.circle(mask, (x, y), r, 255, -1)     # マスク（白塗り）
            break

    x, y, s = calc_centroid(mask)
    print(f"{s=}")
    if x and y:
        return x, y
    else:
        return None

def d_bottle(img):
    # 画像の読み込み
    draw_img = img.copy()
    # HSVに変換
    hsv_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2HSV)

    # 色の閾値設定
    lower_target = np.array([0, 0, 0])
    upper_target = np.array([180, 30, 100]) 

    # 指定範囲に入る画素を抽出（inRangeWrapを使用）
    mask = inRangeWrap(hsv_img, lower_target, upper_target)
    
    # ノイズ処理（モルフォロジー変換）
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 画面内で一番大きい塊だけを抽出する処理
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)
    
    if len(contours) > 0:
        # 面積が一番大きい輪郭を見つける
        max_cnt = max(contours, key=cv2.contourArea)

        # 【修正】変数名を max_cnt に統一し、一定以上の大きさ（面積500以上）なら採用
        if cv2.contourArea(max_cnt) > 500: 
            cv2.drawContours(clean_mask, [max_cnt], -1, 255, thickness=cv2.FILLED)
        else:
            # 大きな塊がない場合はNoneを返す
            return None
    else:
        return None

    try:
        # ノイズ除去後の clean_mask を使って座標を計算する
        result = calc_centroid(clean_mask)
        if result is not None:
            x, y, s = result
            print(f"{s=}")
            return x, y
        else:
            return None
    except TypeError:
        return None