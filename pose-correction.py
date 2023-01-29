# YOLOv7 Yoga-Pose-Correction Tutorial

import cv2
import time
import torch
import argparse
import numpy as np
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from utils.general import non_max_suppression_kpt, strip_optimizer
from torchvision import transforms
from trainer import Check_Angle

# = 1.0= import Libraries==========
import tensorflow
from PIL import ImageFont, ImageDraw, Image
# ==================================


# ====2.0= UI helper function===============
def draw_Ui(img, x1, y1, x2, y2, tx, ty, pose_name, font):
    color = (254, 118, 136)
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)

    draw.rounded_rectangle((x1, y1, x2, y2), fill=color, radius=10)

    draw.text((tx, ty), f"{pose_name}", font=font, fill=(255, 255, 255))

    img = np.array(im)

    return img
# =======================================


@torch.no_grad()
def run(poseweights='yolov7-w6-pose.pt', source='pose.mp4', device='cpu'):

    path = source
    ext = path.split('/')[-1].split('.')[-1].strip().lower()
    if ext in ["mp4", "webm", "avi"] or ext not in ["mp4", "webm", "avi"] and ext.isnumeric():
        input_path = int(path) if path.isnumeric() else path
        device = select_device(opt.device)
        half = device.type != 'cpu'
        model = attempt_load(poseweights, map_location=device)
        _ = model.eval()

        cap = cv2.VideoCapture(input_path)
        webcam = False

        if (cap.isOpened() == False):
            print('Error while trying to read video. Please check path again')

        fw, fh = int(cap.get(3)), int(cap.get(4))
        if ext.isnumeric():
            webcam = True
            fw, fh = 1280, 768
        vid_write_image = letterbox(
            cap.read()[1], (fw), stride=64, auto=True)[0]

        resize_height, resize_width = vid_write_image.shape[:2]
        out_video_name = "output" if path.isnumeric(
        ) else f"{input_path.split('/')[-1].split('.')[0]}"
        out = cv2.VideoWriter(f"./results/{out_video_name}_test1.mp4", cv2.VideoWriter_fourcc(
            *'mp4v'), 30, (resize_width, resize_height))
        if webcam:
            out = cv2.VideoWriter(f"{out_video_name}_mon.mp4", cv2.VideoWriter_fourcc(
                *'mp4v'), 30, (fw, fh))

        frame_count, total_fps = 0, 0

        # =3.0===Load custom font===========
        fontpath = "sfpro.ttf"
        font = ImageFont.truetype(fontpath, 32)
        font1 = ImageFont.truetype(fontpath, 15)
        font2 = ImageFont.truetype(fontpath, 24)
        # ===================================

        # ==4.0=== Load trained pose-indentification model======
        tf_model = tensorflow.keras.models.load_model('model4.h5')
        # ==================================================

        # == 5.0 == variable declaration===========
        sequence = []
        pose_name = ''
        actions = np.array(["Warrior-II-pose", "T-pose", "Tree-pose",
                           "Upward-Facing-Dog", "Downward-Facing-Dog", "Mountain-pose"])
        label_map = {label: num for num, label in enumerate(actions)}
        i = 1
        seq = 30
        # =============================================
        while cap.isOpened:

            # print(f"Frame {frame_count} Processing")
            ret, frame = cap.read()
            if ret:
                orig_image = frame
                # preprocess image
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                if webcam:
                    image = cv2.resize(
                        image, (fw, fh), interpolation=cv2.INTER_LINEAR)
                image = letterbox(image, (fw),
                                  stride=64, auto=True)[0]
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))

                image = image.to(device)
                image = image.float()
                start_time = time.time()

                with torch.no_grad():
                    output, _ = model(image)

                output = non_max_suppression_kpt(
                    output, 0.5, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
                output = output_to_keypoint(output)
                img = image[0].permute(1, 2, 0) * 255
                img = img.cpu().numpy().astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # == 6.0 === preprocess model input data and pose prediction =======
                if i <= seq:
                    for idx in range(output.shape[0]):
                        kpts = output[idx, 7:].T
                        # plot_skeleton_kpts(img, kpts, 3)
                        sequence.append(kpts.tolist())
                if len(sequence) == 30:
                    result = tf_model.predict(np.expand_dims(sequence, axis=0))
                    pose_name = actions[np.argmax(result)]

                if i == seq:
                    sequence = []
                    i = 0
                i += 1
                # =============================================================
                xstart = (fw//2)-150
                ystart = (fh-100)
                yend = (fh-50)

                # = 7.0 == Draw prdiction ==================================
                if pose_name == "T-pose":
                    img = draw_Ui(img, xstart, ystart, xstart+120,
                                  yend, xstart+5, ystart+2, pose_name, font)
                    img = Check_Angle(img, kpts, pose_name, font1, font2)

                elif pose_name == "Tree-pose":
                    img = draw_Ui(img, xstart, ystart, xstart+160,
                                  yend, xstart+5, ystart+2, pose_name, font)
                    img = Check_Angle(img, kpts, pose_name, font1, font2)

                elif pose_name == "Warrior-II-pose":
                    img = draw_Ui(img, xstart, ystart, xstart+247,
                                  yend, xstart+10, ystart+2, pose_name, font)
                    img = Check_Angle(img, kpts, pose_name, font1, font2)

                elif pose_name == "Upward-Facing-Dog":
                    img = draw_Ui(img, xstart, ystart, xstart+320,
                                  yend, xstart+10, ystart+2, pose_name, font)
                    img = Check_Angle(img, kpts, pose_name, font1, font2)


                elif pose_name == "Downward-Facing-Dog":
                    img = draw_Ui(img, xstart, ystart, xstart+350,
                                  yend, xstart+5, ystart+2, pose_name, font)
                    img = Check_Angle(img, kpts, pose_name, font1, font2)

                elif pose_name == "Mountain-pose":
                    img = draw_Ui(img, xstart+5, ystart, xstart+240,
                                  yend, xstart+10, ystart+2, pose_name, font)
                    img = Check_Angle(img, kpts, pose_name, font1, font2)

                # ====================================================================

                # display image
                if webcam:
                    cv2.imshow("Detection", img)
                    key = cv2.waitKey(1)
                    if key == ord('c'):
                        break
                else:
                    img_ = img.copy()
                    img_ = cv2.resize(
                        img_, (960, 540), interpolation=cv2.INTER_LINEAR)
                    cv2.imshow("Detection", img_)
                    cv2.waitKey(1)
                    if key == ord('c'):
                        break

                end_time = time.time()
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1
                out.write(img)
            else:
                break

        cap.release()
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str,
                        default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str,
                        help='path to video or 0 for webcam')
    parser.add_argument('--device', type=str, default='cpu',
                        help='cpu/0,1,2,3(gpu)')

    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device, opt.poseweights)
    main(opt)
