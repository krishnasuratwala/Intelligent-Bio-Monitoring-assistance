Here is the content converted into a clean, structured Markdown format suitable for a GitHub `README.md` or project documentation.

---

# Eye Tracking and Head Pose Data Collection Workflow

### STEP 1: Face Mesh Initialization

Use **MediaPipe** to create **478 facial landmarks** in real-time using a webcam feed.

### STEP 2: Pupil Detection

Using the coordinates from Step 1, detect the exact pupil center and surround the iris with a bounding box using the following landmark indices:

* **Left Eye:** * **Iris/Bounding Box:** 476, 475, 474, 477
* **Pupil Center:** 473


* **Right Eye:** * **Iris/Bounding Box:** 471, 470, 469, 472
* **Pupil Center:** 468



### STEP 3: Relative Positioning

Calculate the difference between the pupil centers and the following specific facial landmarks.

> **Note:** All  and  coordinates must be normalized relative to the frame ( to ) using:
> 1. 
> 2. 
> 
> 

* **Left Pupil Center vs:** `[463, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 389, 251, 284, 332, 297, 338, 10, 109, 67, 103, 53, 21, 162, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 164, 0]`
* **Right Pupil Center vs:** `[33, 7, 163, 144, 153, 154, 155, 145, 133, 173, 157, 158, 159, 160, 161, 246, 389, 251, 284, 332, 297, 338, 10, 109, 67, 103, 53, 21, 162, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 164, 0]`

### STEP 4: Eye Aspect Ratio (EAR)

Calculate the **EAR** values for both eyes. Append these values and the differences calculated in Step 3 to the CSV file.

### STEP 5: Roll Movement Detection

Add the following coordinates to the CSV to assist in detecting head **Roll**:
`[10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 164, 0, 267, 269, 270, 409, 375, 321, 405, 314, 17, 84, 181, 91, 146, 37, 39, 40, 185, 61]`

### STEP 6: Yaw Movement Detection

To detect head **Yaw**, calculate the difference between the following points and save to CSV:

1. 10 and (105, 338)
2. 151 and 0 (67, 297)
3. 9 and (21, 251)
4. 8 and (162, 389)
5. 8 and (162, 389)
6. 6 and (127, 356)
7. 195 and (234, 454)
8. 5 and (93, 366)

### STEP 7: Pitch Movement Detection

To detect head **Pitch**, calculate the difference between landmark **10** and the following set:
`[116, 118, 47, 195, 277, 349, 345]`
Append results to the CSV.

### STEP 8: Image Extraction and Preprocessing

For future data instance mapping:

1. Extract and crop the left and right eye images from the frame.
2. Apply **Histogram Equalization** for enhancement.
3. Include a **5-pixel margin** around the cropped eye images.
4. Save images to a parent directory and store the relative file path in the CSV.

### STEP 9: Data Cleaning/Noise Removal

Implement logic to exclude data instances if:

* The eyes are closed.
* The face is only partially visible in the frame.
* The image/frame is blurry.
* The data instance contains incomplete values.

### STEP 10: Distance Estimation

Calculate the distance between the user’s face and the screen using:

1. **Focal Length Calibration:** 

2. **Distance Calculation:** 


Add distance values to the CSV.

### STEP 11: Cursor Tracking

Implement logic to record the  coordinates of the mouse cursor. The user should look at the cursor constantly while performing various head poses.

### STEP 12: Data Collection Goal

Run the script continuously until a dataset of approximately **30,000 lines** of data is collected.

