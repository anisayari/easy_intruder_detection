##Anis Ayari - Defend Intelligence
# Intrusion detection system

import imutils
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import argparse
from yolov3.models import *
from yolov3.utils.datasets import *
from yolov3.utils.parse_config import *
import smtplib, ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

classes = load_classes('yolov3/'+parse_data_cfg('yolov3/data/coco.data')['names'])
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
number_of_frame_to_pass_after_email_sent = 3000

def send_alert_notification(sender_email, receiver_email, password_email,frame):
    now = datetime.now()

    full_date = now.strftime("%d%m%Y-%H%M%S")
    filename = "intruder_detected_image/intruder-{}.jpg".format(full_date)
    cv2.imwrite(filename,frame)
    date = now.strftime("%d/%m/%Y")
    hour = now.strftime("%H:%M:%S")


    subject = "Intruder detected !"
    body = """
    Hello, an intruder has been detected by your AI system the {0} at {1}. Please find attached a picture of the intruder. Note that the system will be off for a moment.
    """.format(date,hour)
    smtp_server = "smtp.gmail.com"
    port = 465  # For SSL

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message["Bcc"] = receiver_email

    # Add body to email
    message.attach(MIMEText(body, "plain"))


    # Open PDF file in binary mode
    with open(filename, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    # Encode file in ASCII characters to send by email
    encoders.encode_base64(part)

    # Add header as key/value pair to attachment part
    part.add_header(
        "Content-Disposition",
        "attachment; filename= {}".format(filename),
    )

    # Add attachment to message and convert message to string
    message.attach(part)
    text = message.as_string()

    # Log in to server using secure context and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password_email)
        server.sendmail(sender_email, receiver_email, text)
    print('[INFO] email send')
    print('[INFO] System entering sleeping mode for a moment')

def predict_yolo(frame, model, img_size, device):
    nms_thres = 0.5
    conf_thres = 0.3

    img0 = frame
    assert img0 is not None, 'cannot read frame'

    img = letterbox(img0, new_shape=img_size)[0]

    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, np.float32)
    img /= 255.0

    t0 = time.time()
    rects = []

    img = torch.from_numpy(img).to(device)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img)[0]

    pred = non_max_suppression(pred, conf_thres, nms_thres)

    for i, det in enumerate(pred):
        s, im0 = '', img0
        s += '%gx%g ' % img.shape[2:]
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()
                s += '%g %ss, ' % (n, classes[int(c)])
            for *xyxy, conf, _, cls in det:
                if classes[int(cls)] != 'person':
                    continue
                x = xyxy
                rects.append((int(x[0]), int(x[1]), int(x[2]), int(x[3])))

    print('Done. (%.3fs)' % (time.time() - t0))
    return rects

def init_yolo():
    weights,  view_img = 'yolov3/weights/yolov3-spp.weights', 'store_true'
    device = torch_utils.select_device()
    img_size = 416
    model = Darknet('yolov3/cfg/yolov3-spp.cfg', img_size)
    attempt_download(weights)
    if weights.endswith('.pt'):
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:
        _ = load_darknet_weights(model, weights)
    model.to(device).eval()
    return model, img_size, device,  view_img



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--skip-frames", type=int, default=30,
                    help="# of skip frames between detections")
    ap.add_argument("-re", "--receiver-email", type=str,default = '',
                    help="# of skip frames between detections")
    ap.add_argument("-se", "--sender-email", type=str,default = '',
                    help="# of skip frames between detections")
    ap.add_argument("-pe", "--password-email", type=str,default = '',
                    help="# of skip frames between detections")
    args = vars(ap.parse_args())
    receiver_email = args["receiver_email"]
    sender_email = args["sender_email"]
    password_email = args["password_email"]
    if sender_email != '' and receiver_email != '':
        send_email = True
    else:
        send_email = False

    model, img_size, device, view_img = init_yolo()
    writer = None
    W = None
    H = None
    totalFrames = 0
    first_trame = True
    flag = False
    draw_line = False
    print("[INFO] Starting system")
    video_capture = cv2.VideoCapture(0)
    #video_capture = cv2.VideoCapture('video_demo.mp4')
    fps = FPS().start()
    detection = 0
    while True:
        ret, frame = video_capture.read()
        frame = imutils.resize(frame, width=800)
        (H, W) = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        status = "Waiting"

        if totalFrames % args["skip_frames"] == 0:
            rects = []
            status = "Detecting"
            trackers = []
            rects = predict_yolo(frame, model, img_size, device)

        for (x, y, w, h) in rects:
            rect = [x,y,x+w,y+h]
            cv2.rectangle(frame, (rect[0], rect[1]),  (rect[2], rect[3]), (0,0,255), thickness=1)
            cv2.putText(frame, 'intruder', (rect[0], rect[1] - 10), 0, 1, [0,0, 255], thickness=1, lineType=cv2.LINE_AA)

        if detection == 0:
            detection_status = 'Active'
        else:
            detection_status = 'Email system restart in {}/{}'.format(detection,number_of_frame_to_pass_after_email_sent)

        if len(rects)>0 or 0 < detection < 50:
            intruder_status= 'Detected !'
        else:
            intruder_status = 'No Detected'
        info = [
            ("Status", status),
            ('Detection', detection_status),
            ('Intruder', intruder_status)
        ]


        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        totalFrames += 1
        fps.update()
        if len(rects) > 0 and send_email and detection == 0:
            send_alert_notification(sender_email, receiver_email, password_email, frame)
            detection = 1

        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            if v == "Detected !":
                cv2.putText(frame, text, (int(W*0.15), int(H*0.7)),
                            cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)
            else:
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Frame", frame)

        if 0 < detection < number_of_frame_to_pass_after_email_sent:
            detection += 1
        else:
            detection = 0

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    video_capture.release()
    cv2.destroyAllWindows()
