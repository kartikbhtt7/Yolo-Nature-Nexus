from ultralytics import YOLO

def convert_to_onnx(pt_path, output_path):
    model = YOLO(pt_path)
    model.export(format="onnx", imgsz=640, simplify=True, opset=12)
    
    # import os
    # base_name = os.path.basename(pt_path).replace('.pt', '.onnx')
    # os.rename(base_name, output_path)
    # print(f"Model converted to {output_path}")

if __name__ == "__main__":
    convert_to_onnx('models/best_model.pt', 'models/best_model.onnx')