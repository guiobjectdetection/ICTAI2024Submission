from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver.common.by import By
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from ultralytics import YOLO
import cv2
from fuzzywuzzy import process
import time
from paddleocr import PaddleOCR
import config
import uuid
from selenium.webdriver.common.action_chains import ActionChains
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from clip import clip
import cv2
import numpy as np




class SeleniumEyes(Chrome):
    def __init__(self, model_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = YOLO(model_path)
        self.names=self.model.names
        self.ocr= PaddleOCR(lang='en',show_log=False,show_warning=False,use_gpu=True, use_angle_cls=False,cls=False)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
    
    def infer(self, image_path):
        start_time = time.time()
        image = cv2.imread(image_path)
        color = (0, 255, 0)  # Green
        thickness = 2
    
        results = self.model.predict(source=image, save=True, conf=0.3)
        for elt in results:
            boxes = elt.boxes
            objects = [{} for _ in range(len(boxes))]
            for i, c in enumerate(boxes.cls):
                objects[i]["class"] = self.names[int(c)]
    
            box = elt.boxes.xyxy
            boxes_np = box.cpu().numpy()
    
            for i, row in enumerate(boxes_np):
                x1, y1, x2, y2 = row
                # Expand the bounding box by 10 pixels
                x1 = max(int(x1) - 10, 0)
                y1 = max(int(y1) - 10, 0)
                x2 = min(int(x2) + 10, image.shape[1] - 1)
                y2 = min(int(y2) + 10, image.shape[0] - 1)
    
                objects[i]["box"] = (x1, y1, x2, y2)
    
                # Crop the region of interest (ROI) from the original image
                roi = image[y1:y2, x1:x2]
    
                # Convert the ROI to a format suitable for PaddleOCR
                roi_np = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
                if len(roi_np) > 0:
                    # Apply OCR to the ROI
                    ocr_results = self.ocr.ocr(roi_np)
                    # print(ocr_results)
                    # Process OCR results and store them in the 'objects' dictionary
                    texts = []
                    if ocr_results[0]:
                        for line in ocr_results[0]:
                            _, (text, confidence) = line
                            texts.append({"text": text, "confidence": confidence})
    
                    objects[i]["text"] = texts
    
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.4f} seconds")
    
        return objects


    def find_element_visual(self, element_type, search_text, threshold=30):
        """
        Finds the element of a certain type with the highest similarity to the search_text using fuzzy matching.
        
        :param elements: List of UIElement objects.
        :param element_type: The type of the element to match.
        :param search_text: The text to search for within the elements.
        :param threshold: The minimum matching score to consider (default 80).
        :return: The element with the highest similarity score above the threshold or None if no match is found.
        """

        body = driver.find_element(By.TAG_NAME, 'body')
        img_path = "./tmp/"+str(uuid.uuid4())+".png"
        body.screenshot(img_path)
        best_match = None
        highest_score = threshold  # Start with the threshold as the lowest acceptable score
        elements = self.infer(img_path)
        for element in elements:
            print(element)
            if element["class"].lower() == element_type.lower():
                matches = process.extract(search_text, element["text"], limit=None)
                for match in matches:
                    text, score = match
                    if score >= highest_score:
                        highest_score = score
                        best_match = element

        print(best_match)
        return best_match

    def click_element(self, element):
        if element:
            box = element['box']
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            x_click = (x1 + x2) // 2
            y_click = (y1 + y2) // 2

            print(f"Clicking at coordinates: ({x_click}, {y_click})")

            # Scroll element into view
            body = self.find_element(By.TAG_NAME, 'body')
            self.execute_script("arguments[0].scrollIntoView();", body)

            # Option 1: Click using ActionChains with offset
            # action = ActionChains(self)
            # action.move_to_element_with_offset(body, x_click, y_click).click().perform()
            # action.reset_actions()

            # Option 2: Click using execute_script (alternative method)
            self.execute_script("document.elementFromPoint(arguments[0], arguments[1]).click();", x_click, y_click)


    def encode_images(self, images):
        encoded_images = []
        for img in images:
            try:
                image = self.preprocess(Image.fromarray(img)).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image)
                encoded_images.append(image_features)
            except Exception as e:
                print(f"Error encoding image: {e}")
        return torch.cat(encoded_images) if encoded_images else None



    def find_image_clip(self, search_text, threshold=0.2, debug=False):
        """
        Finds the image that best matches the search_text using CLIP.
        
        :param search_text: The text description to search for.
        :param threshold: The minimum similarity score to consider (default 0.2).
        :param debug: If True, saves a debug image with bounding boxes and scores for all candidates (default False).
        :return: A dictionary containing the element information or None if no match is found.
        """
        # Capture the full page screenshot
        img_path = "tmp/" + str(uuid.uuid4()) + ".png"
        self.save_screenshot(img_path)
        
        # Perform object detection to find all images
        elements = self.infer(img_path)
        image_elements = [element for element in elements if element["class"].lower() == "image"]
        
        if not image_elements:
            return None
        
        # Extract images from detected elements
        full_image = cv2.imread(img_path)
        images = [full_image[y1:y2, x1:x2] for element in image_elements for x1, y1, x2, y2 in [element["box"]]]
        
        # Encode images using CLIP
        image_features = self.encode_images(images)
        
        # Encode search text using CLIP
        text = clip.tokenize([search_text]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text)
        
        # Calculate similarities
        similarities = (image_features @ text_features.T).squeeze(1)
        
        # Find the index of the image with the highest similarity
        best_match_index = similarities.argmax().item()
        best_match_score = similarities[best_match_index].item()
        
        # Debug mode: draw bounding boxes and scores for all candidates
        if debug:
            debug_image = full_image.copy()
            for i, element in enumerate(image_elements):
                x1, y1, x2, y2 = element["box"]
                score = similarities[i].item()
                color = (0, 255, 0) if i == best_match_index else (0, 0, 255)
                cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(debug_image, f"{score:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            debug_img_path = f"debug_{search_text.replace(' ', '_')}.png"
            cv2.imwrite(debug_img_path, debug_image)
            print(f"Debug image saved as {debug_img_path}")
        
        # Check if the best match meets the threshold
        if best_match_score > threshold:
            best_match_element = image_elements[best_match_index]
            
            # Create a dictionary with the same structure as find_element_visual
            result = {
                "class": "image",
                "box": best_match_element["box"],
                "text": [{"text": search_text, "confidence": best_match_score}]
            }
            
            return result
        else:
            return None
        
# Usage example
driver = SeleniumEyes(model_path=config.model_path)
driver.set_window_size(1920,1080)
driver.get('https://leetcode.com/')
elt= driver.find_element_visual("link","Explore")
print(elt)
driver.click_element(elt)


# best_match, score = driver.find_image_clip("American flag")

