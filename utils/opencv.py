import cv2


def showImage(*args):
    for index, arg in enumerate(args):
        cv2.imshow("img {}".format(index), arg)

    # 按任意建結束
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def showImages(**kwargs):
    for key in kwargs:
        cv2.imshow("{}".format(key), kwargs[key])

    # 按任意建結束
    cv2.waitKey(0)
    cv2.destroyAllWindows()
