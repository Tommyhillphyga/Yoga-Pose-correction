import cv2
import numpy as np
import math
from PIL import ImageFont, ImageDraw, Image


# takes three points and return the angle between them
def findAngle(image, kpts, p1, p2, p3, draw=True):
    coord = []
    no_kpt = len(kpts) // 3
    for i in range(no_kpt):
        cx, cy = kpts[3*i], kpts[3*i + 1]
        conf = kpts[3 * i + 2]
        coord.append([i, cx, cy, conf])

    points = (p1, p2, p3)

    # Get landmarks
    x1, y1 = coord[p1][1:3]
    x2, y2 = coord[p2][1:3]
    x3, y3 = coord[p3][1:3]

    # calculate the angles
    angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))

    if angle < 0:
        angle += 360

    # draw coordinates
    if draw:
        # cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,255), 4)
        # cv2.line(image, (int(x3), int(y3)), (int(x2), int(y2)), (255,255,255), 4)
        # cv2.circle(image, (int(x1),int(y1)), 10, (255, 255, 255), cv2.FILLED)
        # cv2.circle(image, (int(x1),int(y1)), 20, (235, 235, 235), 8)
        # cv2.circle(image, (int(x2),int(y2)), 10, (255, 255, 255), cv2.FILLED)
        # cv2.circle(image, (int(x2),int(y2)), 20, (235, 235, 235), 8)
        # cv2.circle(image, (int(x3),int(y3)), 10, (255, 255, 255), cv2.FILLED)
        # cv2.circle(image, (int(x3),int(y3)), 20, (235, 235, 235), 8)
        cv2.putText(image, str(int(angle)), (int(x2) - 50, int(y2) + 50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    return int(angle)


def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def styled_findAngle(image, kpts, p1, p2, p3, draw=True):
    coord = []
    no_kpt = len(kpts) // 3
    for i in range(no_kpt):
        cx, cy = kpts[3*i], kpts[3*i + 1]
        conf = kpts[3 * i + 2]
        coord.append([i, cx, cy, conf])

     # Get landmarks
    x1, y1 = coord[p1][1:3]
    x2, y2 = coord[p2][1:3]
    x3, y3 = coord[p3][1:3]

    # calculate the angle
    angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))

    if angle < 0:
        angle += 360

    if draw:
        cv2.line(image, (int(x1), int(y1)),
                 (int(x2), int(y2)), (84, 61, 247), 3, lineType=cv2.LINE_8)
        cv2.line(image, (int(x3), int(y3)),
                 (int(x2), int(y2)), (84, 61, 247), 3, lineType=cv2.LINE_8)
        cv2.circle(image, (int(x1), int(y1)), 12, (84, 61, 247), cv2.FILLED)
        cv2.circle(image, (int(x2), int(y2)), 12, (84, 61, 247), cv2.FILLED)
        cv2.circle(image, (int(x3), int(y3)), 12, (84, 61, 247), cv2.FILLED)


def Check_Angle(img, kpts, pose_name, font, font1):

    right_elbow_angle = findAngle(img, kpts, 6, 8, 10, draw=False)
    left_elbow_angle = findAngle(img, kpts, 5, 7, 9, draw=False)
    right_shoulder_angle = findAngle(img, kpts, 8, 6, 12, draw=False)
    left_shoulder_angle = findAngle(img, kpts, 11, 5, 7, draw=False)
    right_knee_angle = findAngle(img, kpts, 12, 14, 16, draw=False)
    left_knee_angle = findAngle(img, kpts, 11, 13, 15, draw=False)
    right_wrest_angle = findAngle(img, kpts, 6, 12, 14, draw=False)
    left_wrest_angle = findAngle(img, kpts, 5, 11, 13, draw=False)

    cre = False
    cle = False
    crk = False
    clk = False
    crs = False
    cls = False

    icon = cv2.imread("icon.png")
    icon = cv2.resize(icon, (20, 20), interpolation=cv2.INTER_LINEAR)
    color = (254, 118, 136)
    red = (84, 61, 247)
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    draw.rounded_rectangle((50, 80, 280, 370), fill=color, radius=20)

    draw.rounded_rectangle((150, 150, 270, 170), fill=(
        255, 255, 255), radius=10)  # right arm
    draw.rounded_rectangle((150, 180, 270, 200), fill=(
        255, 255, 255), radius=20)  # left arm
    draw.rounded_rectangle((150, 210, 270, 230), fill=(
        255, 255, 255), radius=20)  # right leg
    draw.rounded_rectangle((150, 240, 270, 260), fill=(
        255, 255, 255), radius=20)  # left leg
    draw.rounded_rectangle((200, 270, 270, 290), fill=(
        255, 255, 255), radius=20)  # right shoulder ***
    draw.rounded_rectangle((200, 300, 270, 320), fill=(
        255, 255, 255), radius=20)  # left shoulder ***
    draw.rounded_rectangle((150, 330, 270, 350), fill=(
        255, 255, 255), radius=20)  # wrest

    draw.text((65, 100), f"POSTURE-BOARD", font=font1, fill=(255, 255, 255))
    draw.text((60, 150), f"RIGHT-ARM:", font=font, fill=(255, 255, 255))
    draw.text((60, 180), f"LEFT-ARM:", font=font, fill=(255, 255, 255))
    draw.text((60, 210), f"RIGHT-LEG:", font=font, fill=(255, 255, 255))
    draw.text((60, 240), f"LEFT-LEG:", font=font, fill=(255, 255, 255))
    draw.text((60, 270), f"RIGHT-SHOULDER:", font=font, fill=(255, 255, 255))
    draw.text((60, 300), f"LEFT-SHOULDER:", font=font, fill=(255, 255, 255))
    draw.text((60, 330), f"WREST:", font=font, fill=(255, 255, 255))

# ================================================================================

    if pose_name == "Mountain-pose":
        if right_elbow_angle < 150 or right_elbow_angle > 190:
            cre = True
            draw.text((160, 150), f" Straighten Up", font=font, fill=red)
        else:
            img = np.array(im)
            img[150:170, 200:220] = icon    # right arm
            im = Image.fromarray(img)

        if left_elbow_angle < 150 or left_elbow_angle > 190:
            cle = True
            draw.text((160, 180), f" Straighten Up", font=font, fill=red)
        else:
            img = np.array(im)
            img[180:200, 200:220] = icon  # left arm
            im = Image.fromarray(img)

        if right_knee_angle < 170 or right_knee_angle > 190:
            crk = True
            draw.text((160, 210), f" Straighten Up", font=font, fill=red)
        else:
            img = np.array(im)
            img[210:230, 200:220] = icon  # right leg
            im = Image.fromarray(img)

        if left_knee_angle < 170 or left_knee_angle > 190:
            clk = True
            draw.text((160, 240), f" Straighten Up", font=font, fill=red)
        else:
            img = np.array(im)
            img[240:260, 200:220] = icon  # left leg
            im = Image.fromarray(img)

        img = np.array(im)
        img[330:350, 200:220] = icon     # wrest
        img[300:320, 225:245] = icon  # left shoulder
        img[270:290, 225:245] = icon  # right shoulder
        im = Image.fromarray(img)
# ================================================================================

    if pose_name == "Warrior-II-pose":
        if right_elbow_angle < 165 or right_elbow_angle > 190:
            cre = True
            draw.text((160, 150), f" Straighten Up", font=font, fill=red)
        else:
            img = np.array(im)
            img[150:170, 200:220] = icon    # right arm
            im = Image.fromarray(img)

        if left_elbow_angle < 165 or left_elbow_angle > 190:
            cle = True
            draw.text((160, 180), f" Straighten Up", font=font, fill=red)

        else:
            img = np.array(im)
            img[180:200, 200:220] = icon  # left arm
            im = Image.fromarray(img)

        if right_shoulder_angle > 280 or right_shoulder_angle < 250:
            crs = True
            draw.text((210, 270), f" Adjust", font=font, fill=red)

        else:
            img = np.array(im)
            img[270:290, 225:245] = icon  # right shoulder
            im = Image.fromarray(img)

        if left_shoulder_angle > 280 or left_shoulder_angle < 250:
            cls = True
            draw.text((210, 300), f" Adjust", font=font, fill=red)
        else:
            img = np.array(im)
            img[300:320, 225:245] = icon  # left shoulder
            im = Image.fromarray(img)

# =============  Check knee ============
        if right_knee_angle > 90 and right_knee_angle < 130:
            if right_knee_angle < 130:
                img = np.array(im)
                img[210:230, 200:220] = icon  # right leg
                im = Image.fromarray(img)
            else:
                crk = True
                draw.text((165, 210), f"  Adjust Leg", font=font, fill=red)

        if left_knee_angle > 90 and left_knee_angle < 130:
            if left_knee_angle < 130:
                img = np.array(im)
                img[240:260, 200:220] = icon  # left leg
                im = Image.fromarray(img)
            else:
                clk = True
                draw.text((165, 240), f"  Adjust Leg", font=font, fill=red)

        if left_knee_angle > 195 and left_knee_angle < 230:
            clk = True
            draw.text((160, 240), f"  Adjust Leg", font=font, fill=red)
            img = np.array(im)
            im = Image.fromarray(img)

        if left_knee_angle > 230:
            img = np.array(im)
            img[240:260, 200:220] = icon  # left leg
            im = Image.fromarray(img)

        if right_knee_angle > 150 and right_knee_angle < 170:
            crk = True
            draw.text((160, 210), f" Straighten Up", font=font, fill=red)
            img = np.array(im)
            im = Image.fromarray(img)

        if left_knee_angle > 150 and left_knee_angle < 170:
            clk = True
            draw.text((160, 240), f" Straighten Up", font=font, fill=red)
            img = np.array(im)
            im = Image.fromarray(img)

# ===========
        if left_knee_angle > 170 and left_knee_angle < 195:
            img = np.array(im)
            img[240:260, 200:220] = icon  # left leg
            im = Image.fromarray(img)

        if right_knee_angle > 170 and right_knee_angle < 195:
            img = np.array(im)
            img[210:230, 200:220] = icon  # right leg
            im = Image.fromarray(img)

        img = np.array(im)
        img[330:350, 200:220] = icon     # wrest
        im = Image.fromarray(img)
# =====================================================================================

    if pose_name == "T-pose":
        if right_elbow_angle < 150 or right_elbow_angle > 190:
            cre = True
            draw.text((160, 150), f" Straighten Up", font=font, fill=red)
        else:
            img = np.array(im)
            img[150:170, 200:220] = icon    # right arm
            im = Image.fromarray(img)

        if left_elbow_angle < 150 or left_elbow_angle > 190:
            cle = True
            draw.text((160, 180), f" Straighten Up", font=font, fill=red)
        else:
            img = np.array(im)
            img[180:200, 200:220] = icon  # left arm
            im = Image.fromarray(img)

        if right_knee_angle < 165 or right_knee_angle > 190:
            crk = True
            draw.text((160, 210), f" Straighten Up", font=font, fill=red)
        else:
            img = np.array(im)
            img[210:230, 200:220] = icon  # right leg
            im = Image.fromarray(img)

        if left_knee_angle < 165 or left_knee_angle > 190:
            clk = True
            draw.text((160, 240), f" Straighten Up", font=font, fill=red)
        else:
            img = np.array(im)
            img[240:260, 200:220] = icon  # left leg
            im = Image.fromarray(img)

        img = np.array(im)
        img[330:350, 200:220] = icon     # wrest
        img[300:320, 225:245] = icon  # left shoulder
        img[270:290, 225:245] = icon  # right shoulder
        im = Image.fromarray(img)
# ===============================================================================

    if pose_name == "Tree-pose":
        if right_knee_angle > 100 and right_knee_angle < 195:
            if right_knee_angle > 150 and right_knee_angle < 165:
                crk = True
                draw.text((160, 210), f" Straighten Up", font=font, fill=red)
            if right_knee_angle > 165:
                img = np.array(im)
                img[210:230, 200:220] = icon  # right leg
                im = Image.fromarray(img)
        if right_knee_angle > 100 and right_knee_angle < 150:
            crk = True
            draw.text((165, 210), f"  Adjust Leg", font=font, fill=red)

        if left_knee_angle > 100 and left_knee_angle < 195:
            if left_knee_angle > 150 and left_knee_angle < 165:
                clk = True
                draw.text((160, 240), f" Straighten Up", font=font, fill=red)
            if left_knee_angle > 165:
                img = np.array(im)
                img[240:260, 200:220] = icon  # left leg
                im = Image.fromarray(img)
        if left_knee_angle > 100 and left_knee_angle < 150:
            clk = True
            draw.text((165, 240), f"  Adjust Leg", font=font, fill=red)

        if right_knee_angle > 20 and right_knee_angle < 45:
            if right_knee_angle < 25:
                crk = True
                draw.text((165, 210), f"  Adjust Leg", font=font, fill=red)
            if right_knee_angle > 25 and right_knee_angle < 45:
                img = np.array(im)
                img[210:230, 200:220] = icon  # right leg
                im = Image.fromarray(img)

        if left_knee_angle > 20 and left_knee_angle < 45:
            if left_knee_angle < 25:
                clk = True
                draw.text((165, 240), f"  Adjust Leg", font=font, fill=red)
            if left_knee_angle > 25 and left_knee_angle < 45:
                img = np.array(im)
                img[240:260, 200:220] = icon  # left leg
                im = Image.fromarray(img)

        img = np.array(im)
        img[150:170, 200:220] = icon    # right arm
        img[180:200, 200:220] = icon    # left arm
        img[330:350, 200:220] = icon    # wrest
        img[300:320, 225:245] = icon    # left shoulder
        img[270:290, 225:245] = icon    # right shoulder
        im = Image.fromarray(img)
    # ===========================================================================

    if pose_name == "Upward-Facing-Dog":
        if right_elbow_angle < 170 or right_elbow_angle > 190:
            cre = True
            draw.text((160, 150), f" Straighten Up", font=font, fill=red)
        else:
            img = np.array(im)
            img[150:170, 200:220] = icon    # right arm
            im = Image.fromarray(img)

        if left_elbow_angle < 170 or left_elbow_angle > 190:
            cle = True
            draw.text((160, 180), f" Straighten Up", font=font, fill=red)
        else:
            img = np.array(im)
            img[180:200, 200:220] = icon  # left arm
            im = Image.fromarray(img)

        if right_knee_angle < 165 or right_knee_angle > 190:
            crk = True
            draw.text((160, 210), f" Straighten Up", font=font, fill=red)
        else:
            img = np.array(im)
            img[210:230, 200:220] = icon  # right leg
            im = Image.fromarray(img)

        if left_knee_angle < 165 or left_knee_angle > 190:
            clk = True
            draw.text((160, 240), f" Straighten Up", font=font, fill=red)
        else:
            img = np.array(im)
            img[240:260, 200:220] = icon  # left leg
            im = Image.fromarray(img)

        img = np.array(im)
        img[330:350, 200:220] = icon    # wrest
        img[300:320, 225:245] = icon    # left shoulder
        img[270:290, 225:245] = icon    # right shoulder
        im = Image.fromarray(img)
# ===============================================================================

    if pose_name == "Downward-Facing-Dog":
        if right_elbow_angle < 170 or right_elbow_angle > 190:
            cre = True
            draw.text((160, 150), f" Straighten Up", font=font, fill=red)
        else:
            img = np.array(im)
            img[150:170, 200:220] = icon    # right arm
            im = Image.fromarray(img)

        if left_elbow_angle < 170 or left_elbow_angle > 190:
            cle = True
            draw.text((160, 180), f" Straighten Up", font=font, fill=red)
        else:
            img = np.array(im)
            img[180:200, 200:220] = icon  # left arm
            im = Image.fromarray(img)

        if right_knee_angle < 170 or right_knee_angle > 190:
            crk = True
            draw.text((160, 210), f" Straighten Up", font=font, fill=red)
        else:
            img = np.array(im)
            img[210:230, 200:220] = icon  # right leg
            im = Image.fromarray(img)

        if left_knee_angle < 170 or left_knee_angle > 190:
            clk = True
            draw.text((160, 240), f" Straighten Up", font=font, fill=red)
        else:
            img = np.array(im)
            img[240:260, 200:220] = icon  # left leg
            im = Image.fromarray(img)

        if (left_wrest_angle < 278 or right_wrest_angle < 278):
            draw.text((165, 330), f" Adjust Wrest", font=font, fill=red)
        else:
            img = np.array(im)
            img[330:350, 200:220] = icon    # wrest
            im = Image.fromarray(img)

        img = np.array(im)
        img[300:320, 225:245] = icon    # left shoulder
        img[270:290, 225:245] = icon    # right shoulder
        im = Image.fromarray(img)
# ===============================================================================
    img = np.array(im)
    if cre:
        styled_findAngle(img, kpts, 6, 8, 10)
    if cle:
        styled_findAngle(img, kpts, 5, 7, 9)
    if crk:
        styled_findAngle(img, kpts, 12, 14, 16)
    if clk:
        styled_findAngle(img, kpts, 11, 13, 15)
    if crs:
        styled_findAngle(img, kpts, 8, 6, 12)
    if cls:
        styled_findAngle(img, kpts, 11, 5, 7)

    return img
