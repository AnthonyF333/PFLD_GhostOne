# -*-coding: utf-8 -*-
import os
import onnxruntime
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import cv2

import argparse
import warnings

warnings.filterwarnings("ignore")


def cut_resize_letterbox(image, det, target_size):
    iw, ih = image.size

    facebox_x = det[0]
    facebox_y = det[1]
    facebox_w = det[2] - det[0]
    facebox_h = det[3] - det[1]

    facebox_max_length = max(facebox_w, facebox_h)
    width_margin_length = (facebox_max_length - facebox_w) / 2
    height_margin_length = (facebox_max_length - facebox_h) / 2

    face_letterbox_x = facebox_x - width_margin_length
    face_letterbox_y = facebox_y - height_margin_length
    face_letterbox_w = facebox_max_length
    face_letterbox_h = facebox_max_length

    top = -face_letterbox_y if face_letterbox_y < 0 else 0
    left = -face_letterbox_x if face_letterbox_x < 0 else 0
    bottom = face_letterbox_y + face_letterbox_h - ih if face_letterbox_y + face_letterbox_h - ih > 0 else 0
    right = face_letterbox_x + face_letterbox_w - iw if face_letterbox_x + face_letterbox_w - iw > 0 else 0

    margin_image = Image.new('RGB', (iw + right - left, ih + bottom - top), (0, 0, 0))
    margin_image.paste(image, (left, top))

    face_letterbox = margin_image.crop((face_letterbox_x, face_letterbox_y, face_letterbox_x + face_letterbox_w, face_letterbox_y + face_letterbox_h))
    face_letterbox = face_letterbox.resize(target_size, Image.Resampling.BICUBIC)

    return face_letterbox, facebox_max_length / target_size[0], face_letterbox_x, face_letterbox_y


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def py_cpu_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]  # because index start from 1

    return keep


def process_output(dets, thresh, scale, pad_w, pad_h, iw, ih):
    process_dets = []
    for det in dets:
        if det[4] < thresh:
            continue

        cx, cy, w, h = det[:4]
        x1 = max(((cx - w / 2.) - pad_w) / scale, 0.)
        y1 = max(((cy - h / 2.) - pad_h) / scale, 0.)
        x2 = min(((cx + w / 2.) - pad_w) / scale, iw)
        y2 = min(((cy + h / 2.) - pad_h) / scale, ih)
        score = det[4] * det[15]

        process_dets.append([x1, y1, x2, y2, score])

    return process_dets


def pad_image(image, target_size):
    iw, ih = image.size
    w, h = target_size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale + 0.5)
    nh = int(ih * scale + 0.5)

    pad_w = (w - nw) // 2
    pad_h = (h - nh) // 2
    image = image.resize((nw, nh), Image.Resampling.BICUBIC)
    new_image = Image.new('RGB', target_size, (128, 128, 128))

    new_image.paste(image, (pad_w, pad_h))

    return new_image, scale, pad_w, pad_h


def get_img_tensor(pil_img, use_cuda, target_size, transform):
    iw, ih = pil_img.size
    if iw != target_size[0] or ih != target_size[1]:
        pil_img = pil_img.resize(target_size, Image.Resampling.BICUBIC)

    tensor_img = transform(pil_img)
    tensor_img = torch.unsqueeze(tensor_img, 0)
    if use_cuda:
        tensor_img = tensor_img.cuda()

    return tensor_img


def inference_folder(args):
    if not os.path.exists(args.save_result_folder):
        os.makedirs(args.save_result_folder)

    # Load Models
    facedetect_session = onnxruntime.InferenceSession(args.facedetect_onnx_model)
    pfld_session = onnxruntime.InferenceSession(args.pfld_onnx_model)

    detect_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    pfld_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Inference
    for img_name in os.listdir(args.test_folder):
        img_path = os.path.join(args.test_folder, img_name)
        if '.jpg' not in img_path and '.png' not in img_path:
            continue

        pil_img = Image.open(img_path)
        iw, ih = pil_img.size

        # Detect Face
        pil_img_pad, scale, pad_w, pad_h = pad_image(pil_img, args.facedetect_input_size)
        detect_tensor_img = get_img_tensor(pil_img_pad, args.use_cuda, args.facedetect_input_size, detect_transform)

        inputs = {facedetect_session.get_inputs()[0].name: to_numpy(detect_tensor_img)}
        outputs = facedetect_session.run(None, inputs)
        preds = outputs[0][0]

        preds = np.array(process_output(preds, 0.5, scale, pad_w, pad_h, iw, ih))
        keep = py_cpu_nms(preds, 0.5)
        dets = preds[keep]

        draw = ImageDraw.Draw(pil_img)
        # for det in dets:
        #     draw.rectangle(((det[0], det[1]), (det[2], det[3])), fill=None, outline=(0, 255, 127), width=2)
        # pil_img.save(os.path.join(args.save_result_folder, img_name))

        for det in dets:
            cut_face_img, scale_l, x_offset, y_offset = cut_resize_letterbox(pil_img, det, args.pfld_input_size)

            pfld_tensor_img = get_img_tensor(cut_face_img, args.use_cuda, args.pfld_input_size, pfld_transform)

            inputs = {pfld_session.get_inputs()[0].name: to_numpy(pfld_tensor_img)}
            outputs = pfld_session.run(None, inputs)
            preds = outputs[0][0]

            for i in range(98):
                center_x = preds[i * 2] * args.pfld_input_size[0] * scale_l + x_offset
                center_y = preds[i * 2 + 1] * args.pfld_input_size[1] * scale_l + y_offset
                radius = 1
                draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), (0, 255, 127))

        pil_img.save(os.path.join(args.save_result_folder, img_name.split('.')[0] + '.png'))


def inference_video(args):
    # Load Models
    facedetect_session = onnxruntime.InferenceSession(args.facedetect_onnx_model)
    pfld_session = onnxruntime.InferenceSession(args.pfld_onnx_model)

    detect_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    pfld_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    cap = cv2.VideoCapture(args.test_video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out_cap = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

    prev_pts = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        iw, ih = pil_img.size

        # Detect Face
        pil_img_pad, scale, pad_w, pad_h = pad_image(pil_img, args.facedetect_input_size)
        detect_tensor_img = get_img_tensor(pil_img_pad, args.use_cuda, args.facedetect_input_size, detect_transform)

        inputs = {facedetect_session.get_inputs()[0].name: to_numpy(detect_tensor_img)}
        outputs = facedetect_session.run(None, inputs)
        preds = outputs[0][0]

        preds = np.array(process_output(preds, 0.5, scale, pad_w, pad_h, iw, ih))
        keep = py_cpu_nms(preds, 0.5)
        dets = preds[keep]

        draw = ImageDraw.Draw(pil_img)
        # for det in dets:
        #     draw.rectangle(((det[0], det[1]), (det[2], det[3])), fill=None, outline=(0, 255, 127), width=2)
        # pil_img.save(os.path.join(args.save_result_folder, img_name))

        for det in dets:
            cut_face_img, scale_l, x_offset, y_offset = cut_resize_letterbox(pil_img, det, args.pfld_input_size)

            pfld_tensor_img = get_img_tensor(cut_face_img, args.use_cuda, args.pfld_input_size, pfld_transform)

            inputs = {pfld_session.get_inputs()[0].name: to_numpy(pfld_tensor_img)}
            outputs = pfld_session.run(None, inputs)
            preds = outputs[0][0]

            if len(prev_pts) == 0:
                for i in range(98):
                    center_x = preds[i * 2] * args.pfld_input_size[0] * scale_l + x_offset
                    center_y = preds[i * 2 + 1] * args.pfld_input_size[1] * scale_l + y_offset
                    prev_pts.append(center_x)
                    prev_pts.append(center_y)

            for i in range(98):
                center_x = preds[i * 2] * args.pfld_input_size[0] * scale_l + x_offset
                center_y = preds[i * 2 + 1] * args.pfld_input_size[1] * scale_l + y_offset

                beta = 0.7
                smooth_center_x = center_x * beta + prev_pts[i * 2] * (1 - beta)
                smooth_center_y = center_y * beta + prev_pts[i * 2 + 1] * (1 - beta)

                prev_pts[i * 2] = smooth_center_x
                prev_pts[i * 2 + 1] = smooth_center_y

                radius = 4
                draw.ellipse((smooth_center_x - radius, smooth_center_y - radius, smooth_center_x + radius, smooth_center_y + radius), (0, 255, 127))

        cv_img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
        out_cap.write(cv_img)

    cap.release()
    out_cap.release()


def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--use_cuda', default=True, type=bool)
    parser.add_argument('--pfld_onnx_model', default="./onnx_models/PFLD_GhostOne_112_1_opt_sim.onnx", type=str)
    parser.add_argument('--pfld_input_size', default=(112, 112), type=list)
    parser.add_argument('--facedetect_onnx_model', default="./onnx_models/yolov5face_n_640.onnx", type=str)
    parser.add_argument('--facedetect_input_size', default=(640, 640), type=list)
    parser.add_argument('--test_folder', default='./test_imgs', type=str)
    parser.add_argument('--save_result_folder', default='./test_imgs_result', type=str)
    parser.add_argument('--test_video', default='./test_video/nice.mp4', type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    inference_folder(args)
    # inference_video(args)
