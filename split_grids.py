import os
from PIL import Image

def split_grid(image_path, output_dir, columns=3, rows=2):
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            tile_w = w // columns
            tile_h = h // rows
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            for r in range(rows):
                for c in range(columns):
                    idx = r * columns + c + 1
                    left = c * tile_w
                    top = r * tile_h
                    right = left + tile_w
                    bottom = top + tile_h
                    
                    tile = img.crop((left, top, right, bottom))
                    output_filename = f"{base_name}_part{idx}.png"
                    output_path = os.path.join(output_dir, output_filename)
                    tile.save(output_path)
                    print(f"Saved: {output_filename}")
                
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def main():
    target_dir = r"D:\@home\aero\comfy\ComfyUI\custom_nodes\GridComposite\AlbedoLora\target"
    backup_dir = os.path.join(target_dir, "original_grids")
    
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        
    for filename in os.listdir(target_dir):
        if filename.lower().endswith('.png') and os.path.isfile(os.path.join(target_dir, filename)):
            full_path = os.path.join(target_dir, filename)
            
            # Check size to ensure it's a grid (4608, 3072)
            try:
                # Use with to ensure handle is closed
                with Image.open(full_path) as img:
                    is_correct_size = (img.size == (4608, 3072))
                    size = img.size
                
                if is_correct_size:
                    print(f"Processing grid: {filename}")
                    split_grid(full_path, target_dir)
                    # Move original to backup
                    try:
                        os.rename(full_path, os.path.join(backup_dir, filename))
                    except OSError as e:
                        print(f"Failed to move {filename}: {e}")
                else:
                    if not filename.lower().endswith(('_part1.png', '_part2.png', '_part3.png', '_part4.png', '_part5.png', '_part6.png')):
                        print(f"Skipping {filename}: unexpected size {size}")
            except Exception as e:
                print(f"Could not open {filename}: {e}")

if __name__ == "__main__":
    main()
