# Live Face Identification with pre-trained VGGFace2 model

This is complementary source code for my [blog post](https://www.dlology.com/blog/live-face-identification-with-pre-trained-vggface2-model/).

 Here is the demo

![face identification demo](https://gitcdn.xyz/cdn/Tony607/blog_statics/master/images/face/face_identification.gif "face identification demo")



## Dependencies
- Python3.5
- numpy 1.13.3+mkl
- Keras 2.0.8+
- TensorFlow 1.4.0
- opencv 1.0.1+
- opencv-python 3.3.0+contrib
- keras_vggface
- scipy

Tested on:
- Windows 10 with Tensorflow 1.4.0 GPU

### install requirements
```
pip3 install -r requirements.txt
```

## Run the demo
First run this to generate images from video files and precompute face features
```
python3 face_identify_demo.py
```
Then run the real time demo,
```
python3 face_identify_demo.py
```
For more info, check out my [write up](https://www.dlology.com/blog/live-face-identification-with-pre-trained-vggface2-model/).