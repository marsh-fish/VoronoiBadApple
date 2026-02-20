import moviepy.editor as mp
import cv2
import numpy as np
import glob
import os

def process_frame(frame):
    h, w, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # 1. Get Canny Edges
    edges = cv2.Canny(gray, 100, 200)

    # 2. Get coordinates using the "Step" logic
    y_coords, x_coords = np.where(edges == 255)
    
    # We create a dictionary to keep only one point per 'cell' of size 'grid_size'
    grid_size = 2.5  # Increase this to make the Voronoi shards bigger/less dense
    sampled_points = {}
    
    for x, y in zip(x_coords, y_coords):
        # We 'snap' the coordinates to a grid. 
        # Only the first point found in each grid cell is saved.
        grid_x, grid_y = x // grid_size, y // grid_size
        if (grid_x, grid_y) not in sampled_points:
            sampled_points[(grid_x, grid_y)] = (float(x), float(y))

    points = list(sampled_points.values())

    # Grid Effect
    # side = 40
    # for i in range(0,w, side):
    #     for j in range(0,h, side):
    #         points.append((i, j))

    # 4. Setup Subdiv2D
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(p)

    # 5. Draw Voronoi Facets
    (facets, centers) = subdiv.getVoronoiFacetList([])
    output = np.full(frame.shape, 255, dtype=np.uint8)

    for i in range(len(facets)):
        facet = np.array(facets[i], dtype=np.int32)
        # Draw the cell outline
        cv2.polylines(output, [facet], True, (0, 0, 0), 1)

    return output

def make_outline_video(input_path, output_path="Voronoi BadApple.mp4"):
    clip = mp.VideoFileClip(input_path)
    
    # You can change the resolution of the result
    # clip = clip.resize(height=1080)

    # Apply our 'process_frame' function to every frame in the video
    outline_clip = clip.fl_image(process_frame)
    
    outline_clip.write_videofile(output_path, codec="libx264", audio=True)

# --- Testing Logic ---
def test():
    # 1. Find all files starting with "test" and ending in .jpg, .png, etc.
    # This pattern matches test_input.jpg, test2.png, test_apple.jpeg, etc.
    valid_extensions = ("test*.jpg", "test*.png", "test*.jpeg")
    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(glob.glob(ext))

    if not image_paths:
        print("No images found starting with 'test'!")
        return

    print(f"Found {len(image_paths)} images. Press any key to see the next one.")

    for path in image_paths:
        # Load the image
        img = cv2.imread(path)
        if img is None:
            continue

        # 2. Run the processing (Convert BGR -> RGB -> Process -> BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result_rgb = process_frame(img_rgb)
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

        # 3. Show the result
        # We include the filename in the window title to keep track
        cv2.imshow(f"Original: {os.path.basename(path)}", img)
        cv2.imshow("Processed Result", result_bgr)
        
        print(f"Showing: {path} (Press any key for next)")
        
        # Wait for keypress
        key = cv2.waitKey(0)
        
        # Optional: Close current windows so they don't stack up
        cv2.destroyAllWindows()
        
        # If you press 'q', stop testing
        if key == ord('q'):
            break

    print("Batch testing complete.")

if __name__ == "__main__":
    # test()
    make_outline_video("BadApple.mp4")