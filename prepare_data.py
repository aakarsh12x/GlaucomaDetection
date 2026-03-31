import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

def prepare_glaucoma_dataset(download_dir="yolo_dataset", max_samples_per_split=1000):
    """
    Downloads a public eye disease dataset from Hugging Face using STREAMING 
    to avoid pulling gigabytes of unused data, and prepares it for YOLO.
    """
    print("Downloading dataset from Hugging Face hub with Streaming (bumbledeep/smdg-full-dataset)...")
    dataset_name = "bumbledeep/smdg-full-dataset"
    
    try:
        # Use streaming=True so we don't have to download the 5GB parquet files!
        ds_train = load_dataset(dataset_name, split='train', streaming=True)
        ds_val = load_dataset(dataset_name, split='validation', streaming=True)
    except Exception as e:
        print(f"Failed to fetch dataset '{dataset_name}': {e}")
        return

    splits = {'train': ds_train, 'val': ds_val}
    class_names = ["Normal", "Glaucoma", "Other"] # Fallback mapping if metadata missing
    
    for split_name, dataset_iter in splits.items():
        print(f"Processing '{split_name}' split...")
        
        count = 0
        for item in dataset_iter:
            if count >= max_samples_per_split:
                break
                
            # Find image key
            if 'image' in item:
                image = item['image']
            elif 'img' in item:
                image = item['img']
            else:
                image = item[list(item.keys())[0]]
                
            # Find label key
            label_col = 'label' if 'label' in item else ('diagnosis' if 'diagnosis' in item else list(item.keys())[-1])
            label_idx = item[label_col]
            
            class_name = class_names[label_idx].replace(" ", "_") if isinstance(label_idx, int) else str(label_idx)
            
            # Format and save image
            if isinstance(image, Image.Image):
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            
                os.makedirs(os.path.join(download_dir, split_name, class_name), exist_ok=True)
                save_path = os.path.join(download_dir, split_name, class_name, f"img_{count}.jpg")
                image.save(save_path)
            
            count += 1
            if count % 10 == 0:
                print(f"  Downloaded {count}/{max_samples_per_split} images for {split_name}...")
                
    print(f"\nDataset successfully prepared with streaming in '{os.path.abspath(download_dir)}'!")

if __name__ == "__main__":
    prepare_glaucoma_dataset()
