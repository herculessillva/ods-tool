# Object Detection and Save Tool

1. First, download YOLO files [here](https://drive.google.com/file/d/1VUSA5fyPf9hwG3c_HJMindCyt_k8DFu1/view?usp=sharing) and extract.

2. Install python dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Clone Darknet repository:
    ```
    git clone https://github.com/AlexeyAB/darknet.git
    ```

4. Generate libso to darknet python wrapper:
    ```
    cd darknet/
    ```
    4.1. Compile for GPU use:
    ```
    make GPU=1 CUDNN=1 LIBSO=1 ARCH='-gencode arch=compute_86,code=[sm_86,compute_86]'
    ```
    4.2. Compile for CPU use:
    ```
    make LIBSO=1
    ```

5. Set your images folder path in ```in_path``` variable, and ```OUT_PATH``` to save in.

6. Run tool:
    ```
    python crop_save.py
    ```