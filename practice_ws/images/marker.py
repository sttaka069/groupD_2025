#!/usr/bin/env python3
# coding: utf-8
from cv2 import aruco, imread, imwrite
import numpy as np

# 引数nのIDのマーカーを作成する
def make_marker(n):
    # Size and offset value
    size = 150
    offset = 10
    x_offset = y_offset = int(offset) // 2

    # get dictionary and generate image
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    ar_img = aruco.generateImageMarker(dictionary, n, size)

    # make white image
    img = np.zeros((size + offset, size + offset), dtype=np.uint8)
    img += 255

    # overlap image
    img[y_offset:y_offset + ar_img.shape[0], x_offset:x_offset + ar_img.shape[1]] = ar_img
    imwrite(f"/root/practice_ws/images/marker_{n}.png", img)

def d_marker(img, n: int):
    # 【追加機能】探しているIDを表示
    print(f"--- [Debug] 探しているマーカーID: {n} ---")

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()

    corners, ids, _ = aruco.detectMarkers(img, dictionary, parameters=parameters)
    
    # 【追加機能】見つかったID一覧を表示
    if ids is not None:
        detected_ids = np.ravel(ids)
        print(f"--- [Debug] カメラに映っているID: {detected_ids} ---")
    else:
        print("--- [Debug] マーカーは何も映っていません ---")

    # 判定処理
    if ids is not None and n in np.ravel(ids):
        index = np.where(ids == n)[0][0]
        cornerUL = corners[index][0][0]
        cornerBR = corners[index][0][2]
        
        center = [ (cornerUL[0]+cornerBR[0])/2 , (cornerUL[1]+cornerBR[1])/2 ]
        print(f"--- [Debug] ID:{n} を発見！座標を返します ---")
        return tuple(center)

    print(f"--- [Debug] ID:{n} は見つかりませんでした... ---")
    return None