from model_load import load_lama_cleaner, load_yolo
from util import get_mask, norm_img
from PIL import Image
import cv2
import numpy as np
import torch



# 변수에 이미지를 받아서 yolo 추론에 넣어야함
def yolo_inference(image_path):
    # yolo 추론
    model = load_yolo()
    results = model(image_path)
    
    boxes = results[0].boxes
    
    xywhn = []
    # class가 '0' 인 bounding box 좌표 리스트에 저장
    
    for i in range(len(boxes)):
        h_class = boxes.cls[i].cpu().numpy()
        if h_class == 0.0:
            xywhn.append(boxes[i].xywhn.cpu().numpy().tolist())
            
    # 3차원 list -> 2차원 list 변환
    flattened_2d_list = [item for sublist in xywhn for item in sublist]
    
    # 2차원 list -> 소수점 6자리 까지 반올림
    rounded_list = [[round(val, 6) for val in sublist] for sublist in flattened_2d_list]
    
    # 최종 좌표 초기화
    final_xywhn = []

    # 최종 좌표 list로 저장
    for result in rounded_list:
        final_xywhn.append(result)

    # print(final_xywhn)
    
    return final_xywhn

def lama_cleaner(image: np.ndarray, mask: np.ndarray, device: str):
    model = load_lama_cleaner()
    
    image = norm_img(image)
    mask = norm_img(mask)

    mask = (mask > 0) * 1
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    mask = torch.from_numpy(mask).unsqueeze(0).to(device)

    inpainted_image = model(image, mask) # inference

    cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
    cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")

    return Image.fromarray(cur_res)

# Web Service 에서는 Image를 input 시 DB에 원본 이미지를 저장하고 그 경로를 가져와서 추론 / mask 이미지 / lama 추론 실행
# test 입장에서는 직접적인 경로를 활용
def main():
    # 이미지 파일 경로
    image_path = '../images/img1.jpg'
    # 이미지 로드
    image = Image.open(image_path)
    # 이미지를 NumPy 배열로 변환 (선택 사항)
    image_array = np.array(image)
    image_array = cv2.resize(image_array, (904, 1552))
    # yolo 추론
    boxes = yolo_inference(image_path)
    
    # box 변환 후 마스크 get
    get_mask_image = get_mask(boxes, image_array)
    # lama 추론
    # import pdb
    # pdb.set_trace()
    print(image_array.shape)
    print(get_mask_image.shape)
    yolo_lama_cleaner = lama_cleaner(image_array, get_mask_image, device='cuda')
    yolo_lama_cleaner.save('./result.png')

if __name__ == "__main__":

    # TODO: 이미지, 마스크 변수 초기화 후 실행
    # image = np.array()
    # mask = np.array()
    # device='cuda'
    
    # lama_cleaner(image=image, mask=mask, device=device)
    main()