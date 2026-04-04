from ultralytics import YOLO

def check_accuracy():
    print("Loading model/best.pt...")
    model = YOLO('model/best.pt')
    print("Running validation on data/retail_combined.yaml...")
    metrics = model.val(data='data/retail_combined.yaml', workers=0, verbose=False)
    
    print("\n--- Model Accuracy Metrics ---")
    print(f"Mean Average Precision (mAP50-95): {metrics.box.map:.4f}")
    print(f"Mean Average Precision (mAP50):    {metrics.box.map50:.4f}")
    print("\nClass-wise mAP50:")
    names = model.names
    
    for i, cls_id in enumerate(metrics.box.ap_class_index):
        if cls_id < len(names):
            print(f"  {names[cls_id]}: {metrics.box.maps[i]:.4f}")

if __name__ == '__main__':
    check_accuracy()
