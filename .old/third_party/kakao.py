import json

import cv2
import requests
import sys

LIMIT_BOX = 40
# 4cdbada9320e35dfc7730e4e750b5e58

def kakao_ocr_detect(image_path: str, appkey: str):
    """
    detect api request example
    :param image_path: 이미지파일 경로
    :param appkey: 카카오 앱 REST API 키
    """
    API_URL = 'https://kapi.kakao.com/v1/vision/text/detect'

    headers = {'Authorization': 'KakaoAK {}'.format(appkey)}

    image = cv2.imread(image_path)
    jpeg_image = cv2.imencode(".jpg", image)[1]
    data = jpeg_image.tobytes()

    return requests.post(API_URL, headers=headers, files={"file": data})


def kakao_ocr_recognize(image_path: str, boxes: list, appkey: str):
    """
    recognize api request example
    :param boxes: 감지된 영역 리스트. Canvas 좌표계: 좌상단이 (0,0) / 우상단이 (limit,0)
                    감지된 영역중 좌상단 점을 기준으로 시계방향 순서, 좌상->우상->우하->좌하
                    ex) [[[0,0],[1,0],[1,1],[0,1]], [[1,1],[2,1],[2,2],[1,2]], ...]
    :param image_path: 이미지 파일 경로
    :param appkey: 카카오 앱 REST API 키
    """
    API_URL = 'https://kapi.kakao.com/v1/vision/text/recognize'

    headers = {'Authorization': 'KakaoAK {}'.format(appkey)}

    image = cv2.imread(image_path)
    jpeg_image = cv2.imencode(".jpg", image)[1]
    data = jpeg_image.tobytes()

    return requests.post(API_URL, headers=headers, files={"file": data}, data={"boxes": json.dumps(boxes)})


def main():
    if len(sys.argv) != 3:
        print(f"Please run with args: $ python {sys.argv[0]} /path/to/image appkey")
        exit(0)
        
    image_path, appkey = sys.argv[1], sys.argv[2]

    output = kakao_ocr_detect(image_path, appkey).json()
    print("[detect] output:\n{}\n".format(output))

    boxes = output["result"]["boxes"]
    boxes = boxes[:min(len(boxes), LIMIT_BOX)]
    output = kakao_ocr_recognize(image_path, boxes, appkey).json()
    print("[recognize] output:\n{}\n".format(json.dumps(output, sort_keys=True, indent=2)))


if __name__ == "__main__":
    main()