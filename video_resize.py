import cv2


def video_resize(video_name):
    cap = cv2.VideoCapture(video_name)
    video_name_temp = video_name.strip(".mp4") + "_1280_720" + ".avi"
    videowriter = cv2.VideoWriter(video_name_temp, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 22,
                                  (1280, 720))
    success, _ = cap.read()
    while success:
        success, img1 = cap.read()
        try:
            img = cv2.resize(img1, (1280, 720), interpolation=cv2.INTER_LINEAR)
            videowriter.write(img)
        except:
            break
    return video_name_temp


if __name__ == '__main__':
    video_resize('F:/海浪测量/数据集/production ID_5192710.mp4')