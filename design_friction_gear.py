# ライブラリ
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import cv2
from PIL import Image
import os
import glob
import winsound

# 定数
eq_list = ["exp", "sin3", "sin4", "sin9"]
k  = 6  # 形状のパラメータ
a  = 30 # 中心間距離

# 出力設定
N_theta = 360 # 角度の分解能
imgfmt = ".png" # 画像ファイル形式

# 映像の出力設定
i_value_max = 5
N_rot = 3
framerate = 120
savefmt = ".mp4" # ".mp4" or ".gif"
savename="output_"


# 設計方程式（形状によって変える）
def func_i_exp(theta):
    return np.exp(theta/k) / ( 1 + np.exp(2*np.pi/k) - np.exp(theta/k) )

def func_i_sin4(theta):
    return 1 + np.sin(4*theta) / 3

def func_i_sin3(theta):
    return 1 + 2 * np.sin(3*theta) / 3

def func_i_sin9(theta):
    return 3 + np.sin(9*theta)


# 従動側回転角の計算式（形状によって変える）
def func_phi_exp(theta):
    return k*np.log( np.exp(2*np.pi/k) / ( 1 + np.exp(2*np.pi/k) - np.exp(theta/k) ) )

def func_phi_sin4(theta):
    return theta + ( 1 - np.cos(4*theta) )/12

def func_phi_sin3(theta):
    return theta + ( 1 - np.cos(3*theta) ) * 2/9
    
def func_phi_sin9(theta):
    return 3 * theta + ( 1 - np.cos(9*theta) )/9


def notice2(): # 処理完了通知2
    winsound.Beep(1308,200)
    winsound.Beep(1960,200)
    winsound.Beep(2616,200)
    
# 一般的な関係式
def func_r(i_value):
    r_theta = a * i_value / (i_value + 1)
    r_phi   = a           / (i_value + 1)
    return (r_theta, r_phi)

# 座標変換
def pole2axis(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return  (x, y)



# メイン処理
def main():
    
    for eq in eq_list:
        # フォルダの設定
        print("eq: "+ str(eq))
        main_path = os.getcwd()
        img_path = main_path + '\img_' + eq
        os.makedirs(img_path, exist_ok=True) # フォルダがない場合は作る
        
        # 関数の設定
        if eq == "exp":
            func_i   = func_i_exp
            func_phi = func_phi_exp
        elif eq == "sin3":
            func_i   = func_i_sin3
            func_phi = func_phi_sin3
        elif eq == "sin4":
            func_i   = func_i_sin4
            func_phi = func_phi_sin4
        elif eq == "sin9":
            func_i   = func_i_sin9
            func_phi = func_phi_sin9
        
        # グラフの画像データを生成
        
        d_theta = 2 * np.pi / N_theta
        columns_name = ["theta","x_theta","y_theta","phi","x_phi","y_phi"]
        os.chdir(img_path)
        for j_theta_rot in range(N_theta):
            
            print("\rmaking images %d / %d"%(int(j_theta_rot+1), int(N_theta)), end="")
            
            # 回転角の大きさ
            theta_rot = 0
            phi_rot = func_phi(theta_rot)
            for j_theta in range(j_theta_rot):
                theta_rot = 2 * np.pi / N_theta * j_theta_rot
                phi_rot = func_phi(theta_rot)
            i_value_rot = func_i(theta_rot)
            
            # 座標の計算
            df    = pd.DataFrame([], columns=columns_name)
            for j_theta in range(N_theta+1):
                theta   = 2 * np.pi / N_theta * j_theta
                phi     = func_phi(theta)
                i_value = func_i(theta)
                r_theta, r_phi   = func_r(i_value)
                x_theta, y_theta = pole2axis(r_theta, theta-theta_rot  )
                x_phi,   y_phi   = pole2axis(r_phi,   np.pi-phi+phi_rot)
                
                df0 = pd.DataFrame([[theta, x_theta, y_theta, phi, x_phi, y_phi]], columns=columns_name)
                df  = pd.concat([df,df0])
            
            df0 = df.head(1) # 輪郭線を一周させる
            df  = pd.concat([df,df0])
            
            df["x_phi"] = df["x_phi"] + a # 噛み合うように従動側を平行移動する
            
            # プロット
            # https://qiita.com/simonritchie/items/da54ff0879ad8155f441
            fig = plt.figure(figsize=(10, 5))
            masterspec = gridspec.GridSpec(ncols=2, nrows=1,width_ratios=[10, 1])
            subspec = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=masterspec[1],height_ratios=[1, 100])
            
            ax0 = fig.add_subplot(masterspec[0])
            plt.plot(df["x_theta"],df["y_theta"], color="orange")
            plt.plot(df["x_phi"  ],df["y_phi"  ], color="blue")
            plt.plot([0,a],[0,0],"o",color="black")
            plt.xlim(-a,a*2)
            plt.ylim(-a,a)
            plt.grid()
            ax0.set_aspect('equal', adjustable='box')
            
            ax1 = fig.add_subplot(subspec[1])
            plt.bar([0], [i_value_rot], align="center", color="blue")           # 中央寄せで棒グラフ作成
            plt.xticks([0], ["angular velocity ratio"])  # X軸のラベル
            plt.ylim(0, i_value_max) # np.ceil(max(df_i["i_value"]))                     
            
            # 画像として出力
            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            cv2.imwrite("img_gear_"+str(j_theta_rot).zfill(8)+imgfmt,img)
            
            plt.clf()
            plt.close()
        
        # 画像データの読み込み
        print("\nloading images")
        imgs_singlerot = []
        imgnamelist = sorted(glob.glob("*"+imgfmt)) # 画像ファイルの検索

        for j in range(0,len(imgnamelist)):
            img = cv2.imread(imgnamelist[j]) # 画像の読み込み
            imgs_singlerot.append(img)

        height, width, _ = img.shape
        size = (width, height)
        
        imgs = imgs_singlerot.copy()
        for k in range(N_rot-1):  # 回転数分だけ複製する
                imgs += imgs_singlerot


        # 動画の出力
        print("writing movie")
        os.chdir(main_path)
        if savefmt == '.mp4':
            movie = cv2.VideoWriter(savename+eq+savefmt, cv2.VideoWriter_fourcc(*'MP4V'), framerate, size) # 映像の出力設定
            for i in range(len(imgs)):
                movie.write(imgs[i]) # 映像の書き出し
            movie.write(np.zeros_like(imgs[0])) # なぜか最後が切れるので追加
            movie.release() # 映像の書き出しの終了
            
        elif savefmt == '.gif':
            imgs_PIL = [Image.fromarray(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)) for i in range(len(imgs))]
            imgs_PIL[0].save(savename+eq+savefmt,save_all=True, append_images=imgs_PIL[1:], optimize=False, loop=0)
        else:
            print("save format: "+savefmt+" is wrong.")
        
        notice2()


main()




