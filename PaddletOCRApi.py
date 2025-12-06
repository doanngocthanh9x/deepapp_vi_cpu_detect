import os
import sys
from typing import List, Dict, Tuple, Union, Optional
from warnings import filterwarnings
import cv2
import numpy as np

# Import from existing modules
from nets import nn
from utils import util

filterwarnings("ignore")


class PaddleOCRProcessor:
    """
    Class xử lý OCR sử dụng PaddleOCR models
    Hỗ trợ 2 chức năng chính:
    1. Detect và recognize tất cả text trong ảnh
    2. Recognize text từ bbox cụ thể
    """
    
    def __init__(self, weights_dir: str = 'weights'):
        """
        Khởi tạo PaddleOCRProcessor
        
        Args:
            weights_dir: Đường dẫn đến folder chứa các file weights ONNX
        """
        self.weights_dir = weights_dir
        
        # Check if model files exist
        detection_path = self._resource_path(f'{weights_dir}/detection.onnx')
        recognition_path = self._resource_path(f'{weights_dir}/recognition.onnx')
        classification_path = self._resource_path(f'{weights_dir}/classification.onnx')
        
        if not os.path.exists(detection_path):
            raise FileNotFoundError(f"Detection model not found at {detection_path}")
        if not os.path.exists(recognition_path):
            raise FileNotFoundError(f"Recognition model not found at {recognition_path}")
        if not os.path.exists(classification_path):
            raise FileNotFoundError(f"Classification model not found at {classification_path}")
        
        # Initialize models
        print(f"Initializing PaddleOCRProcessor with weights from '{weights_dir}'...: {detection_path}")
        self.detection = nn.Detection(detection_path)
        self.recognition = nn.Recognition(recognition_path)
        self.classification = nn.Classification(classification_path)
        
        print(f"✓ PaddleOCRProcessor initialized successfully!")
        print(f"  - Weights directory: {weights_dir}")
    
    def _resource_path(self, relative_path: str) -> str:
        """Get absolute path to resource, works for dev and for PyInstaller"""
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(os.path.dirname(__file__), relative_path)
    
    def process_full_image(self, image: Union[str, np.ndarray]) -> Dict:
        """
        Xử lý toàn bộ ảnh để detect và recognize tất cả text
        
        Args:
            image: Đường dẫn ảnh hoặc numpy array
            
        Returns:
            Dict chứa:
            - 'texts': List các text đã nhận dạng
            - 'confidences': List độ tin cậy tương ứng
            - 'bboxes': List các bbox (polygon points)
            - 'count': Số lượng text regions được tìm thấy
        """
        # Load image if path is provided
        if isinstance(image, str):
            frame = cv2.imread(image)
        else:
            frame = image.copy()
        
        if frame is None:
            raise ValueError("Cannot load image")
        
        # Convert color space for processing
        rgb_frame = frame.copy()
        cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB, rgb_frame)  # inplace conversion
        
        # Step 1: Detection
        points = self.detection(rgb_frame)
        points = util.sort_polygon(list(points))
        
        if len(points) == 0:
            return {
                'texts': [],
                'confidences': [],
                'bboxes': [],
                'count': 0
            }
        
        # Step 2: Crop images from detected regions
        cropped_images = [util.crop_image(rgb_frame, x) for x in points]
        
        # Step 3: Classification (angle correction)
        cropped_images, angles = self.classification(cropped_images)
        
        # Step 4: Recognition
        texts, confidences = self.recognition(cropped_images)
        
        # Prepare result
        result = {
            'texts': texts,
            'confidences': confidences,
            'bboxes': points,
            'count': len(texts)
        }
        
        return result
    
    def process_bbox(self, image: Union[str, np.ndarray], bbox: List, bbox_format: str = "polygon") -> Dict:
        """
        Xử lý text trong một bbox cụ thể
        
        Args:
            image: Đường dẫn ảnh hoặc numpy array
            bbox: Bbox coordinates
            bbox_format: Format của bbox:
                - "polygon": List of points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                - "xyxy": [x1, y1, x2, y2]
                - "yolo": [x_center, y_center, width, height] (normalized 0-1)
                
        Returns:
            Dict chứa:
            - 'text': Text đã nhận dạng
            - 'confidence': Độ tin cậy
            - 'bbox': Bbox đã xử lý
        """
        # Load image if path is provided
        if isinstance(image, str):
            frame = cv2.imread(image)
        else:
            frame = image.copy()
        
        if frame is None:
            raise ValueError("Cannot load image")
        
        # Convert bbox to polygon format if needed
        polygon_bbox = self._convert_bbox_to_polygon(bbox, bbox_format, frame.shape)
        
        # Convert color space for processing
        rgb_frame = frame.copy()
        cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB, rgb_frame)
        
        try:
            # Crop image from bbox
            # Convert polygon_bbox to numpy array for crop_image function
            polygon_array = np.array(polygon_bbox, dtype=np.float32)
            cropped_image = util.crop_image(rgb_frame, polygon_array)
            
            # Classification (angle correction)
            cropped_images, angles = self.classification([cropped_image])
            corrected_image = cropped_images[0]
            
            # Recognition
            texts, confidences = self.recognition([corrected_image])
            
            text = texts[0] if len(texts) > 0 else ""
            confidence = confidences[0] if len(confidences) > 0 else 0.0
            
            return {
                'text': text,
                'confidence': confidence,
                'bbox': polygon_bbox
            }
            
        except Exception as e:
            print(f"Error processing bbox: {e}")
            return {
                'text': "",
                'confidence': 0.0,
                'bbox': polygon_bbox
            }
    
    def _convert_bbox_to_polygon(self, bbox: List, bbox_format: str, img_shape: Tuple) -> List:
        """
        Chuyển đổi bbox sang format polygon
        
        Args:
            bbox: Bbox coordinates
            bbox_format: Format của bbox
            img_shape: Shape của ảnh (height, width, channels)
            
        Returns:
            Polygon points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        img_height, img_width = img_shape[:2]
        
        if bbox_format.lower() == "polygon":
            return bbox
        
        elif bbox_format.lower() == "xyxy":
            x1, y1, x2, y2 = bbox
            return [
                [x1, y1],  # top-left
                [x2, y1],  # top-right
                [x2, y2],  # bottom-right
                [x1, y2]   # bottom-left
            ]
        
        elif bbox_format.lower() == "yolo":
            x_center, y_center, width, height = bbox
            
            # Convert normalized to absolute coordinates
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height
            
            # Calculate corners
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            return [
                [x1, y1],  # top-left
                [x2, y1],  # top-right
                [x2, y2],  # bottom-right
                [x1, y2]   # bottom-left
            ]
        
        else:
            raise ValueError(f"Unsupported bbox_format: {bbox_format}")
    
    def convert_paddle_bbox_to_yolo(self, paddle_bbox: List[List[float]]) -> List[float]:
        """
        Chuyển đổi bbox từ format PaddleOCR (polygon) sang YOLO format (xyxy)
        
        Args:
            paddle_bbox: Bbox in polygon format [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            
        Returns:
            Bbox in YOLO format [x1, y1, x2, y2]
        """
        if len(paddle_bbox) != 4:
            raise ValueError("Invalid paddle bbox format. Expected 4 points.")
        
        x_coords = [point[0] for point in paddle_bbox]
        y_coords = [point[1] for point in paddle_bbox]
        
        x1 = min(x_coords)
        y1 = min(y_coords)
        x2 = max(x_coords)
        y2 = max(y_coords)
        
        return [x1, y1, x2, y2]
    
    def process_multiple_bboxes(
        self, 
        image: Union[str, np.ndarray], 
        bboxes: List[List], 
        bbox_format: str = "polygon"
    ) -> List[Dict]:
        """
        Xử lý nhiều bboxes cùng lúc
        
        Args:
            image: Đường dẫn ảnh hoặc numpy array
            bboxes: List các bbox
            bbox_format: Format của bbox
            
        Returns:
            List of dict, mỗi dict chứa kết quả cho một bbox
        """
        results = []
        
        for i, bbox in enumerate(bboxes):
            try:
                result = self.process_bbox(image, bbox, bbox_format)
                result['bbox_index'] = i
                results.append(result)
            except Exception as e:
                print(f"Error processing bbox {i}: {e}")
                results.append({
                    'text': "",
                    'confidence': 0.0,
                    'bbox': bbox,
                    'bbox_index': i
                })
        
        return results
    
    def visualize_results(
        self, 
        image: Union[str, np.ndarray], 
        results: Union[Dict, List[Dict]], 
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Vẽ kết quả lên ảnh
        
        Args:
            image: Ảnh gốc
            results: Kết quả từ process_full_image hoặc process_multiple_bboxes
            show_confidence: Có hiển thị confidence không
            
        Returns:
            Ảnh đã vẽ kết quả
        """
        # Load image if path is provided
        if isinstance(image, str):
            vis_image = cv2.imread(image)
        else:
            vis_image = image.copy()
        
        # Handle single result dict or list of results
        if isinstance(results, dict):
            # Result from process_full_image
            texts = results.get('texts', [])
            confidences = results.get('confidences', [])
            bboxes = results.get('bboxes', [])
        else:
            # Results from process_multiple_bboxes
            texts = [r.get('text', '') for r in results]
            confidences = [r.get('confidence', 0.0) for r in results]
            bboxes = [r.get('bbox', []) for r in results]
        
        # Draw results
        for i, (text, confidence, bbox) in enumerate(zip(texts, confidences, bboxes)):
            if not text.strip():
                continue
                
            # Convert bbox to numpy array for drawing
            bbox_np = np.array(bbox, dtype=np.int32)
            
            # Draw polygon
            cv2.polylines(vis_image, [bbox_np], True, (0, 255, 0), 2)
            
            # Prepare text label
            if show_confidence:
                label = f"{text} ({confidence:.2f})"
            else:
                label = text
            
            # Draw text
            x, y, w, h = cv2.boundingRect(bbox_np)
            cv2.putText(
                vis_image, label, 
                (int(x), int(y - 2)), 
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (200, 200, 0), 1, cv2.LINE_AA
            )
        
        return vis_image


# Example usage
if __name__ == '__main__':
    # Initialize processor
    processor = PaddleOCRProcessor('weights')
    
    print("\n" + "="*60)
    print("DEMO: PaddleOCRProcessor")
    print("="*60)
    
    # Test image path
    test_image_path = r"/home/vps1/WorkSpace/DetectApi/01HM00012302_300005_image_78.png"
    
    print("\n1. FULL IMAGE PROCESSING:")
    print("-" * 40)
    try:
        # Process full image
        full_results = processor.process_full_image(test_image_path)
        
        print(f"✓ Found {full_results['count']} text regions")
        for i, (text, conf) in enumerate(zip(full_results['texts'], full_results['confidences'])):
            # Handle case where conf might be a list
            conf_value = conf[0] if isinstance(conf, list) and len(conf) > 0 else conf
            conf_value = float(conf_value) if conf_value is not None else 0.0
            print(f"  Text {i+1}: '{text}' (confidence: {conf_value:.3f})  (bbox: {full_results['bboxes'][i]})")
    
    except Exception as e:
        print(f"✗ Error in full image processing: {e}")
        # Create dummy image for testing
        print("Creating dummy image for testing...")
        dummy_image = np.ones((200, 400, 3), dtype=np.uint8) * 255
        cv2.putText(dummy_image, "TEST OCR", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        full_results = processor.process_full_image(dummy_image)
        print(f"✓ Found {full_results['count']} text regions in dummy image")
        for i, (text, conf) in enumerate(zip(full_results['texts'], full_results['confidences'])):
            # Handle case where conf might be a list
            conf_value = conf[0] if isinstance(conf, list) and len(conf) > 0 else conf
            conf_value = float(conf_value) if conf_value is not None else 0.0
            print(f"  Text {i+1}: '{text}' (confidence: {conf_value:.3f}) (bbox: {full_results['bboxes'][i]})")
    
    print("\n2. BBOX PROCESSING:")
    print("-" * 40)
    
    # Example bbox (XYXY format)
    example_bbox_xyxy = [50, 50, 200, 100]
    
    try:
        bbox_result = processor.process_bbox(dummy_image, example_bbox_xyxy, "xyxy")
        print(f"XYXY bbox result: '{bbox_result['text']}' (confidence: {bbox_result['confidence']:.3f})")
    except Exception as e:
        print(f"✗ Error in bbox processing: {e}")
    
    # Example bbox (YOLO format)
    example_bbox_yolo = [0.25, 0.25, 0.3, 0.1]  # normalized
    
    try:
        yolo_result = processor.process_bbox(dummy_image, example_bbox_yolo, "yolo")
        print(f"YOLO bbox result: '{yolo_result['text']}' (confidence: {yolo_result['confidence']:.3f})")
    except Exception as e:
        print(f"✗ Error in YOLO bbox processing: {e}")
    
    print("\n" + "="*60)
    print("✓ Demo completed!")
    print("="*60)
    
    print("\n📝 Usage Examples:")
    print("-" * 40)
    print("""
# 1. Khởi tạo processor
processor = PaddleOCRProcessor('weights')

# 2. Xử lý toàn bộ ảnh
results = processor.process_full_image('image.jpg')
print(f"Found {results['count']} texts:")
for text, conf in zip(results['texts'], results['confidences']):
    print(f"  '{text}' (confidence: {conf:.2f})")

# 3. Xử lý bbox cụ thể
# XYXY format
result = processor.process_bbox(image, [x1, y1, x2, y2], "xyxy")
print(f"Text: {result['text']}")

# YOLO format  
result = processor.process_bbox(image, [x_center, y_center, w, h], "yolo")
print(f"Text: {result['text']}")

# Polygon format
result = processor.process_bbox(image, [[x1,y1], [x2,y2], [x3,y3], [x4,y4]], "polygon")
print(f"Text: {result['text']}")

# 4. Xử lý nhiều bboxes
results = processor.process_multiple_bboxes(image, bboxes_list, "xyxy")

# 5. Visualize kết quả
vis_image = processor.visualize_results(image, results)
cv2.imshow('Results', vis_image)
""")
