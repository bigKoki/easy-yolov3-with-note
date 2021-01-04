import torch
import cv2
import numpy as np
from models.yolov3 import yolov3
from utils.nms import non_maximum_suppression
from config.yolov3 import cfg

class demo(object):
    def __init__(self):
        self.model = yolov3().to(torch.device(cfg.device))
        self.load_weights()

    def test(self,path):
        origin_img = cv2.imread(path)
        img_tensor = self.prepare_image(origin_img.copy())

        pred = self.predict(img_tensor)
        pred = pred.cpu().numpy()
        pred = pred[pred[..., 4] > cfg.conf_thresh]
        x1 = pred[..., 0] - pred[..., 2] * 0.5
        y1 = pred[..., 1] - pred[..., 3] * 0.5
        x2 = pred[..., 0] + pred[..., 2] * 0.5
        y2 = pred[..., 1] + pred[..., 3] * 0.5
        pred[..., 0] = x1
        pred[..., 1] = y1
        pred[..., 2] = x2
        pred[..., 3] = y2
        index = non_maximum_suppression(pred, cfg.nms_thresh)
        pred = pred[index]
        h, w, _ = origin_img.shape
        pred[:, 0] = pred[:, 0] * w / cfg.test_input_size
        pred[:, 1] = pred[:, 1] * h / cfg.test_input_size
        pred[:, 2] = pred[:, 2] * w / cfg.test_input_size
        pred[:, 3] = pred[:, 3] * h / cfg.test_input_size
        class_names = self.get_class_names()
        demo_image = origin_img.copy()
        for rec in pred:
            class_ind = np.argmax(rec[5:])
            demo_image = cv2.rectangle(origin_img, (int(rec[0]), int(rec[1])), (int(rec[2]), int(rec[3])),
                                       (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            demo_image = cv2.putText(demo_image, class_names[int(class_ind)], (int(rec[0]), int(rec[1])), font, 1, (255, 255, 255), 2)
        cv2.imshow("result", demo_image)
        cv2.waitKey(0)

    def predict(self,image):
        with torch.no_grad():
            _, pred = self.model(image)
        pred = torch.cat((pred[0].view(-1, 5 + cfg.num_classes), pred[1].view(-1, 5 + cfg.num_classes),
                          pred[2].view(-1, 5 + cfg.num_classes)), dim=0)
        return pred

    def prepare_image(self,image):
        image = cv2.resize(image, (cfg.test_input_size, cfg.test_input_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image.transpose([2, 0, 1])
        image = image / 255.
        image = torch.from_numpy(image).to(torch.device(cfg.device))
        image = image.unsqueeze(0)
        return image

    def load_weights(self):
        checkpoint = torch.load(cfg.weight_path)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def get_class_names(self):
        class_names = []
        with open(cfg.class_path) as class_file:
            classes = class_file.readlines()
            for class_name in classes:
                class_names.append(class_name[:-1])
        return class_names

if __name__ == '__main__':
    image_path = "./data/test/P0003.png"
    tester = demo()
    tester.test(image_path)