import cv2
from ultralytics import YOLO
import math

classNames = ['standing', 'waving', 'waving_and_walking']
model = YOLO("best.pt")

def main():
    cap = cv2.VideoCapture(0)

    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        ret, frame = cap.read()
        results = model(frame, stream=True)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 =box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)


                #put box in camera
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                #confidence box
                confidence = math.ceil((box.conf[0]*100))/100
                print(f"Confidence: {confidence}")

                #class name
                cls = int(box.cls[0])
                print(f"{classNames[cls]}")

                #object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)


            cv2.imshow('test cam', frame)
        
        #exit program press 'q'
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
