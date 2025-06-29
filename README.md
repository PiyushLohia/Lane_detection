
# 🛣️ Road Lane Detection Project

This project demonstrates lane detection on road images and videos using Python, OpenCV, and MoviePy. The pipeline highlights white and yellow lane lines, detects edges, and draws lines using the Hough Transform.

## 📂 Project Structure

```
Road_Lane_Detection/
├── Road_Lane_Detection.py       # Main Python script
├── test_images/                 # Folder with sample input images
│   ├── image1.jpg
│   └── ...
├── test_videos/                 # (Optional) Folder to place input videos
├── solidWhiteRight.mp4          # Sample video file (place here)
├── solidYellowLeft.mp4          # Sample video file (place here)
├── solidWhiteRight_output.mp4   # Output video after processing
├── solidYellowLeft_output.mp4   # Output video after processing
└── README.md                    # Project documentation
```

## ⚙️ Features

- White and yellow lane line detection using color thresholding
- Support for multiple color spaces: RGB, HSV, HSL
- Edge detection with Canny algorithm
- Region of interest masking
- Line detection using Hough Transform
- Works with both images and video files

## 🧪 Example Results

| Input Frame | Output with Detected Lanes |
|-------------|-----------------------------|
| ![](test_images/image1.jpg) | ![](output_images/image1_output.jpg) |

*(Replace with actual result images)*

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/road-lane-detection.git
   cd road-lane-detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Or install manually:**
   ```bash
   pip install numpy matplotlib opencv-python moviepy
   ```

## ▶️ Usage

### 🖼️ For images:
Just run the script. All `.jpg` files from the `test_images/` folder will be processed and displayed.

### 📹 For videos:
Make sure the video files `solidWhiteRight.mp4` and `solidYellowLeft.mp4` are in the same directory as the script. Then run:

```bash
python Road_Lane_Detection.py
```

This will generate:
- `solidWhiteRight_output.mp4`
- `solidYellowLeft_output.mp4`

## 🧾 Notes

- Ensure the input videos exist before running.
- You can customize kernel sizes, thresholds, or region of interest polygon for better accuracy.
- Default test videos can be downloaded from [Udacity's GitHub Repo](https://github.com/udacity/CarND-LaneLines-P1/tree/master/test_videos)

## 📸 Sample Visualization

The script uses `matplotlib` to visualize each stage:
- Color filtering
- Grayscale conversion
- Edge detection
- ROI masking
- Hough line drawing

## 👨‍💻 Author

**Piyush Lohia**  
📍 Agra, Uttar Pradesh  
📧 piyushlohia1857@gmail.com  

## 📄 License

This project is for educational purposes. You may adapt it freely with attribution.
