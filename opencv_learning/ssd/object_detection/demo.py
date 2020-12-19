import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.autograd import Variable
from opencv_learning.ssd.data import VOC_CLASSES as labels
from opencv_learning.ssd import buildSSD
from opencv_learning.ssd.data import BaseTransform, VOC_CLASSES as labelmap

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# initialize SSD
net = buildSSD('test', 300, 21)
net = net.cuda()
print(net)

net.load_weights('data/SSD/ssd300_mAP_77.43_v2.pth')
transform = BaseTransform(net.size, (104 / 256.0, 117 / 256.0, 123 / 256.0))


def detect(frame, net, transform):
    height, width = frame.shape[:2]
    frame_t = transform(frame)[0]

    # (n_channel, height, width)
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    x = x.unsqueeze(0).cuda()
    y = net(x)

    detections = y.data
    scale = torch.tensor([width, height, width, height])
    # detections = [batch, number of classes, number of occurence, (score, x0, Y0, x1, y1)]
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),
                        2, cv2.LINE_AA)
            j += 1
    return frame


def detectTester():
    image = Image.open('data/image/golden-retriever-puppy.jpg')
    # plt.imshow(image)
    # plt.show()

    # image = image.resize((300,300), Image.ANTIALIAS)
    image = np.asarray(image)
    rgb_image = np.asarray(image)

    # region 由 BaseTransform 取代此段功能
    # '''先Resize 成 300*300'''
    # x = cv2.resize(image, (300, 300)).astype(np.float32)
    #
    # '''Vgg 16的預處理方式'''
    # x -= (104.0, 117.0, 123.0)
    #
    # '''將type轉乘np.float32'''
    # x = x.astype(np.float32)
    #
    # '''將 RGB 轉換為 BGR '''
    # x = x[:, :, ::-1].copy()

    # plt.imshow(x)
    # plt.imshow(x_hat)
    # plt.show()

    '''轉換成torch 的 tensor，並且將 H,W,C 轉成 C,H,W'''
    # x = torch.from_numpy(x).permute(2, 0, 1)
    # endregion

    '''要先用Variable包裝才能送給Pytorch模型'''
    # region 由 detect(frame, net, transform) 取代此段功能
    # xx = x.unsqueeze(0)
    # if torch.cuda.is_available():
    #     xx = xx.cuda()
    #
    # '''Forward Pass'''
    # y = net(xx)
    #
    # # torch.Size([1, 21, 200, 5])
    # '''Batch Size, 類別數, top 200 個框, 5=(delta x, delta y, delta h, delta w, confidence)'''
    # # print(y.shape)
    #
    # plt.figure(figsize=(10, 10))
    # colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    # plt.imshow(rgb_image)  # plot the image for matplotlib
    # currentAxis = plt.gca()
    #
    # detections = y.data
    # # scale each detection back up to the image
    # '''得到『寬、高、寬、高』的值，藉此將x1,y1,x2,y2回放為原本尺寸'''
    # scale = torch.tensor(rgb_image.shape[1::-1]).repeat(2)
    # for i in range(detections.size(1)):
    #     j = 0
    #     '''信心程度 > 0.6 的預測框才計算'''
    #     while detections[0, i, j, 0] >= 0.6:
    #         score = detections[0, i, j, 0]
    #         label_name = labels[i - 1]
    #         display_txt = '%s: %.2f' % (label_name, score)
    #         pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
    #         coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
    #         color = colors[i]
    #         currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
    #         currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})
    #         j += 1
    # endregion

    dst = detect(image, net, transform)
    cv2.imshow('detect', dst)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def realTimeDetect():
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        canvas = detect(frame, net, transform)
        cv2.imshow('Video', canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def videoCaptureTester():
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# detectTester()
# realTimeDetect()
videoCaptureTester()
