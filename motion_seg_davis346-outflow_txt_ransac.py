import glob
import os
import json
import numpy as np
import cv2
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans
from PIL import Image



def draw_events(save_path, evs_num, idx, evs_w, evs_h, evs_x, evs_y, evs_p, color):
    if color == 'GRAY':
        img = np.ones(shape=(evs_h, evs_w), dtype=np.uint8) * 0
        for j in range(evs_num):
            # img[y1[j], x1[j]] = (2*p1[j]-1)  # p is [0, 1], convert it to [-1,1], only keep last p
            img[evs_y[j], evs_x[j]] = 255
        image = img

        img1 = np.ones(shape=(evs_h, evs_w), dtype=int) * 0.5
        for j in range(evs_num):
            img1[evs_y[j], evs_x[j]] += (2 * evs_p[j] - 1) * 0.25  # p is [0, 1], convert it to [-0.25, 0.25], only keep last p; img: [0.5, 0.75(positive), 0.25(negative)]
        # convert img to red & blue map
        tmp0 = (img1 * 255).astype(np.uint8)
        tmp1 = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        rgbArray = np.zeros((tmp1.shape[0], tmp1.shape[1], tmp1.shape[2]), 'uint8')
        tmp1[tmp1 == 127] = 0
        B = tmp0.copy()
        G = tmp0.copy()
        R = tmp0.copy()

        B[tmp0 > 127] = 0
        B[tmp0 <= 127] = 255
        rgbArray[:, :, 0] = B

        G[tmp0 != 127] = 0
        G[tmp0 == 127] = 255
        rgbArray[:, :, 1] = G

        R[tmp0 >= 127] = 255
        R[tmp0 < 127] = 0
        rgbArray[:, :, 2] = R
        # image1 = rgbArray.astype(np.uint8)

        image1 = Image.new('RGB', (evs_w, evs_h), color='white')
        for i in range(evs_num):
            if evs_p[i] == 1:
                image1.putpixel((evs_x[i], evs_y[i]), (255, 0, 0))
            else:
                image1.putpixel((evs_x[i], evs_y[i]), (0, 0, 255))
        image1 = np.array(image1)
        image1 = image1[:, :, (2, 1, 0)]

    elif color == 'RB':
        img = np.ones(shape=(evs_h, evs_w), dtype=int) * 0.5
        for j in range(evs_num):
            img[evs_y[j], evs_x[j]] += (2 * evs_p[j] - 1) * 0.25  # p is [0, 1], convert it to [-0.25, 0.25], only keep last p; img: [0.5, 0.75(positive), 0.25(negative)]
        # convert img to red & blue map
        tmp0 = (img * 255).astype(np.uint8)
        tmp1 = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        rgbArray = np.zeros((tmp1.shape[0], tmp1.shape[1], tmp1.shape[2]), 'uint8')
        tmp1[tmp1 == 127] = 0
        B = tmp0.copy()
        G = tmp0.copy()
        R = tmp0.copy()

        B[tmp0 > 127] = 0
        B[tmp0 <= 127] = 255
        rgbArray[:, :, 0] = B

        G[tmp0 != 127] = 0
        G[tmp0 == 127] = 255
        rgbArray[:, :, 1] = G

        R[tmp0 >= 127] = 255
        R[tmp0 < 127] = 0
        rgbArray[:, :, 2] = R
        image = rgbArray.astype(np.uint8)
    elif color == 'TimeImage':
        # 速度太慢，目前都是用的遍历
        img = np.zeros(shape=(evs_h, evs_w), dtype=int)
        threshold = np.zeros(shape=(evs_h, evs_w), dtype=int)
        for n in range(evs_h):
            for m in range(evs_w):
                pixel_num = len(np.intersect1d(np.argwhere(evs_x == m),np.argwhere(evs_y == n)))
                
                img[n, m] = pixel_num * 30
                if pixel_num * 30 > 100:
                    threshold[n, m] = 255
                else:
                    threshold[n, m] == 0
                # print(pixel_num)
        image = img
        name = str(idx).zfill(5)
        # thresh = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite(os.path.join(save_path, name + '_seg.png'), threshold)

        

    name = str(idx).zfill(5)
    cv2.imwrite(os.path.join(save_path, name + '.png'), image1)
    return image ,image1

def generate_event_frame(evs_w, evs_h, evs_x, evs_y, evs_p):
    img = np.ones(shape=(evs_h, evs_w), dtype=int) * 0.5

    temp = (2 * evs_p - 1) * 0.25
    img[evs_y, evs_x] += temp

    # for j in range(evs_num):
    #     img[evs_y[j], evs_x[j]] += (2 * evs_p[j] - 1) * 0.25  # p is [0, 1], convert it to [-0.25, 0.25], only keep last p; img: [0.5, 0.75(positive), 0.25(negative)]
    # convert img to red & blue map
    tmp0 = (img * 255).astype(np.uint8)
    tmp1 = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    rgbArray = np.zeros((tmp1.shape[0], tmp1.shape[1], tmp1.shape[2]), 'uint8')
    tmp1[tmp1 == 127] = 0
    B = tmp0.copy()
    G = tmp0.copy()
    R = tmp0.copy()

    B[tmp0 > 127] = 0
    B[tmp0 <= 127] = 255
    rgbArray[:, :, 0] = B

    G[tmp0 != 127] = 0
    G[tmp0 == 127] = 255
    rgbArray[:, :, 1] = G

    R[tmp0 >= 127] = 255
    R[tmp0 < 127] = 0
    rgbArray[:, :, 2] = R
    image = rgbArray.astype(np.uint8)

    return image



# 主要函数
def event_frame(aedat_path, events, h, w, slice_time, duration_us, num_events, event_name, save_txtfile, save_h5file, ev_color, imus, delta_imu, threshold, n_clusters):
    # 路径设置
    if slice_time:
        frame_path = os.path.join(os.path.dirname(aedat_path), os.path.basename(aedat_path).replace('.aedat4', '') + '_t=' + str(duration_us) + 'us' +"_ransac"+ '_threshold=' + str(threshold))
    else:
        frame_path = os.path.join(os.path.dirname(aedat_path), os.path.basename(aedat_path).replace('.aedat4', '') + '_n=' + str(num_events))
    frame_save_path = os.path.join(frame_path, 'event_frame')
    frame_cal_save_path = os.path.join(frame_path, 'event_frame_calib')
    time_image_save_path = os.path.join(frame_path, 'time_image')
    seg_image_save_path = os.path.join(frame_path, 'seg')
    class_image_save_path = os.path.join(frame_path, 'class')
    cluster_image_save_path = os.path.join(frame_path, 'cluster')
    event_frame_3D_save_path = os.path.join(frame_path, 'event_frame_3D')

    event_compen_txt_save_path = os.path.join(frame_path, 'event_compen_txt_save_path')
    seg_npy_save_path = os.path.join(frame_path, 'seg_npy_save_path')


    if not os.path.exists(frame_save_path):
        os.makedirs(frame_save_path)
    if not os.path.exists(frame_cal_save_path):
        os.makedirs(frame_cal_save_path)
    if not os.path.exists(time_image_save_path):
        os.makedirs(time_image_save_path)
    if not os.path.exists(seg_image_save_path):
        os.makedirs(seg_image_save_path)
    if not os.path.exists(class_image_save_path):
        os.makedirs(class_image_save_path)
    
    if not os.path.exists(cluster_image_save_path):
        os.makedirs(cluster_image_save_path)


    if not os.path.exists(event_frame_3D_save_path):
        os.makedirs(event_frame_3D_save_path)

    if not os.path.exists(event_compen_txt_save_path):
        os.makedirs(event_compen_txt_save_path)

    if not os.path.exists(seg_npy_save_path):
        os.makedirs(seg_npy_save_path)

    # Access information of all events by type
    # 简化后的DAVIS346内参数
    focus = 6550
    pixel_size = 18.5

    imu_temp = imus[:, 1:4]

    t1 = events[:, 0]
    x1 = events[:, 1].astype(int)
    y1 = events[:, 2].astype(int)
    p1 = events[:, 3]

    x_origin = events[:, 1].astype(int)
    y_origin = events[:, 2].astype(int)
    t_origin = events[:, 0]
    p_origin = events[:, 3]

    name = event_name


    #保存原始事件帧
    origin_event_image = generate_event_frame(346, 260, x_origin, y_origin, p_origin)



    cv2.imwrite(os.path.join(frame_save_path, name + '.png'), origin_event_image)




    average_gyro = np.mean(imu_temp, axis=0)
    # 这一块都是运动补偿，矩阵计算方式，实时性加速
    if True:
        time_diff = (t1 - t1[0]) / 1000000.0

        x_rotation_angles = average_gyro[0] * time_diff
        y_rotation_angles = average_gyro[1] * time_diff
        z_rotation_angles = average_gyro[2] * time_diff

        x_angular = np.deg2rad(x_rotation_angles)
        y_angular = np.deg2rad(y_rotation_angles)
        z_angular = np.deg2rad(z_rotation_angles)

        # 目标像素坐标->目标图像坐标系
        x_temp = x_origin.astype(np.int16) - w/2
        y_temp = y_origin.astype(np.int16) - h/2
        # 内参引起的变换角度
        pre_x_angel = np.arctan(y_temp*pixel_size/focus)
        pre_y_angel = np.arctan(x_temp*pixel_size/focus)
        # 目标图像坐标->参考帧像素坐标系（注意，这里没有位移，仅有旋转）
        compen_x = (x_temp*np.cos(z_angular) - np.sin(z_angular)*y_temp) - (x_temp - (focus*np.tan(pre_y_angel + y_angular)/pixel_size)) + w/2
        compen_y = (x_temp*np.sin(z_angular) + np.cos(z_angular)*y_temp) - (y_temp - (focus*np.tan(pre_x_angel - x_angular)/pixel_size)) + h/2

        x1 = compen_x
        y1 = compen_y

    # 删除越界的事件(传感器记录错误)
    delete_id = np.vstack((np.argwhere(x1 > w-1), np.argwhere(x1 < 0), np.argwhere(y1 > h-1), np.argwhere(y1 < 0)))
    x1 = np.delete(x1, delete_id).astype(int)
    y1 = np.delete(y1, delete_id).astype(int)
    p1 = np.delete(p1, delete_id)
    t1 = np.delete(t1, delete_id)
    evs_duration_num = len(x1)
    # print(evs_duration_num)
    # x1 = x_origin.astype(np.int)
    # y1 = y_origin.astype(np.int)
    # p1 = p_origin.astype(np.int)
    # t1 = t_origin.astype(np.int)

    # 统计图像（count image）计算
    img_color = np.ones(shape=(h, w), dtype=int) * 0.5
    img_conf = np.zeros(shape=(h, w), dtype=float)
    count_img = np.zeros(shape=(h, w), dtype=float)
    for j in range(evs_duration_num):
        img_color[y1[j], x1[j]] += (2 * p1[j] - 1) * 0.25  # p is [0, 1], convert it to [-0.25, 0.25], only keep last p; img: [0.5, 0.75(positive), 0.25(negative)]
        img_conf[y1[j], x1[j]] +=  t1[j] # p is [0, 1], convert it to [-0.25, 0.25], only keep last p; img: [0.5, 0.75(positive), 0.25(negative)]
        count_img[y1[j], x1[j]] += 1
    mask = np.where(count_img > 0, 1, 0)


    # 运动补偿后的事件帧
    tmp0 = (img_color * 255).astype(np.uint8)
    tmp1 = cv2.cvtColor((img_color * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    rgbArray = np.zeros((tmp1.shape[0], tmp1.shape[1], tmp1.shape[2]), 'uint8')
    tmp1[tmp1 == 127] = 0
    B = tmp0.copy()
    G = tmp0.copy()
    R = tmp0.copy()

    B[tmp0 > 127] = 0
    B[tmp0 <= 127] = 255
    rgbArray[:, :, 0] = B

    G[tmp0 != 127] = 0
    G[tmp0 == 127] = 255
    rgbArray[:, :, 1] = G

    R[tmp0 >= 127] = 255
    R[tmp0 < 127] = 0
    rgbArray[:, :, 2] = R
    image = rgbArray.astype(np.uint8)

    image_compen = Image.new('RGB', (w, h), color='white')
    for i in range(len(x1)):
        if p1[i] == 1:
            image_compen.putpixel((x1[i], y1[i]), (255, 0, 0))
        else:
            image_compen.putpixel((x1[i], y1[i]), (0, 0, 255))
    image_compen1 = np.array(image_compen)

    image_compen1 = image_compen1[:, :, (2, 1, 0)]
    cv2.imwrite(os.path.join(frame_cal_save_path, name + '.png'), image_compen1)

    #保存补偿事件流
    data_compen_event_txt = np.column_stack((t1, x1, y1, p1))
    np.savetxt(os.path.join(event_compen_txt_save_path, name + ".txt"), data_compen_event_txt, fmt="%.32f %d %d %d")

    #保存补偿事件流3D图

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0))  # X轴背景颜色为透明
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0))  # Y轴背景颜色为透明
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0))  # Z轴背景颜色为透明
    ax.grid(False)
    colors = np.where(p1 == 1, 'red', 'blue')
    ax.scatter(x1, y1, t1, c=colors, marker='o', s=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('t')
    plt.savefig(os.path.join(event_frame_3D_save_path, name + '.png'), bbox_inches='tight')




    # 时间图像计算
    if True:
        # 使用时间图,改成矩阵求概率图形式存储为时间概率图
        confidence_map = np.zeros(shape=(h, w), dtype=np.float64)
        T_Average = np.mean(t1)
        T_map = img_conf / (count_img + 0.000001)
        confidence_map = ((T_map-T_Average*mask)/(duration_us) + mask * np.ones(shape=(h, w), dtype=np.float64)) / 2.
        # confidence_map = (T_map - T_Average * mask) / (duration_us)
        image_norm =(confidence_map * 255).astype(np.uint8)
        color_image = cv2.applyColorMap(image_norm, cv2.COLORMAP_HOT)

        cv2.imwrite(os.path.join(time_image_save_path, name + '.png'), color_image)
        # print("时间图像")

        # 运动分类：背景区域和潜在独立运动区域，此处的threshold后面修改为自适应方式
        B[confidence_map > threshold] = 255
        B[confidence_map <= threshold] = 0
        B[confidence_map == 0] = 0

        G[confidence_map > threshold] = 160
        G[confidence_map <= threshold] = 255
        G[confidence_map == 0] = 0

        R[confidence_map > threshold] = 255
        R[confidence_map < 0] = 0
        R[confidence_map <= threshold] = 0

        rgbArray[:, :, 0] = B
        rgbArray[:, :, 1] = G
        rgbArray[:, :, 2] = R
        class_map = rgbArray.astype(np.uint8)
        # 形态学操作，暂时不计入，会导致边缘模糊
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # 5x5的矩形结构元素
        # seg_map = cv2.erode(seg_map, kernel, iterations=1)
        # seg_map = cv2.dilate(seg_map, kernel, iterations=1)
        cv2.imwrite(os.path.join(class_image_save_path, name + '.png'), class_map)
        # print(confidence_map.shape,"confidence_map")
        # 潜在动态区域提取分割、滤波
        binary_image = np.where(confidence_map > threshold, 255, 0).astype(np.uint8)
        binary_image = cv2.medianBlur(binary_image.astype(np.uint8), ksize=3)

        # 保存分割图与npy文件
        np.save(os.path.join(seg_npy_save_path, name + '.npy'),binary_image)
        cv2.imwrite(os.path.join(seg_image_save_path, name + '.png'), binary_image)

        # 聚类（我是采用的Kmeans聚类，可以用别的），分别从空间和运动两个维度进行聚类
        cluster_image = (binary_image / 255.) * image_norm

        n_cluster = n_clusters


        # 特征构建（从这后面开始开发）
        features = cluster_image.reshape(w * h, 1)



        kmeans = KMeans(n_clusters=n_cluster, random_state=0,n_init=10)
        kmeans.fit(features)
        # #  获取每个像素点的聚类标签
        #
        labels = kmeans.labels_
        label_image = labels.reshape(h, w)

        # 分割图上色
        label_id = 1
        B[label_image == label_id] = 255
        B[label_image != label_id] = 0

        G[label_image == label_id] = 255
        G[label_image != label_id] = 0

        R[label_image == label_id] = 255
        R[label_image != label_id] = 0

        rgbArray[:, :, 0] = B
        rgbArray[:, :, 1] = G
        rgbArray[:, :, 2] = R
        seg_map = rgbArray.astype(np.uint8)

        #
        #
        cv2.imwrite(os.path.join(cluster_image_save_path, name + '.png'), seg_map)

















def read_file_davis(ev_fold,imu_fold, formate, slice_time, dt, n, split_evs, start_ts, end_ts, s_img, s_txt, s_h5, event_color, delta_imu, threshold, n_clusters):

    height = 260
    width = 346

    event_txt_path = glob.glob(os.path.join(ev_fold, "*.txt"))
    imu_txt_path = glob.glob(os.path.join(imu_fold, "*.txt"))

    for i in range(len(event_txt_path)):
        events = np.loadtxt(event_txt_path[i])
        imus = np.loadtxt(imu_txt_path[i])
        event_name = os.path.basename(event_txt_path[i]).replace('.txt', '')

        event_frame(ev_fold, events, height, width, slice_time, dt, n, event_name, s_txt, s_h5, event_color, imus, delta_imu, threshold, n_clusters)

if __name__ == '__main__':
    events_color = 'GRAY'  # events color(red(+) and blue(-)), or 'GRAY'
    data_formate = 'frame'
    # data_formate = 'voxel'

    split_events = False  # 按照[split_start_ts, split_end_ts]的时间区间提取事件
    split_start_ts = 10000000  # us
    split_end_ts = 70000000  # us

    slice_by_time = True  # 使用固定时间生成事件帧，False为使用固定数量间隔生成事件帧
    n_events = 10000  # 生成事件帧的事件数量

    imu_delta = 0.001 # 单位s
    save_img = True
    save_txt = False
    save_h5 = False
    IMG_TYPE = 'RB'

    # 可调节的参数就这几个，其他不要随意调
    file_path = r'K:\ICRA2023_code\motion_seg_for_github\test_data'




    event_txt_fold = os.path.join(file_path, "event_txt")
    imu_txt_fold = os.path.join(file_path, "imu_txt")

    delta_t = 40000  # 生成事件帧每个片段的间隔时间(us)
    seg_threshold = 0.6 # 分割阈值（这个可根据实际情况调试，到时候后续加入根据IMU值来自适应设置）
    n_clusters = 3 # 没有事件区域：0，前景区域：1， 背景区域：2



    read_file_davis(event_txt_fold,imu_txt_fold,data_formate, slice_by_time, delta_t, n_events, split_events, split_start_ts,  split_end_ts,
                    save_img, save_txt, save_h5, events_color, imu_delta, seg_threshold, n_clusters)
    print('done')
