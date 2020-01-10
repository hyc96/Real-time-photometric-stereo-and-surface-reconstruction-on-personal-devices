# Real-time-photometric-stereo-and-surface-reconstruction-on-personal-devices

Photometric stereo is a method that estimates an object’s surface depth and orientation from images of the same view taken under different light directions. Conceptually, three images with distinct light directions are sufficient in obtaining the surface normal map. However, noise is a prevalent factor in many facets of this process and industrial practice often require multiple images taken in a controlled environment. We propose to explore the robustness of shape from shading algorithms and adapt them such that real-time surface reconstruction can be achieved on personal devices. This entails using minimal input data and computation. In following sections we show that we retrieved accurate 3D surfaces of objects in real-time from a webcam live video, using only light sources from a computer screen.


## Prerequisites

- cv2
- screeninfo
- mayavi
- mpl_toolkits


## Usage 

Run 
```
$python main.py 
```
and input as prompted.

## Example

- Sample object:

![alt text](https://github.com/hyc96/Real-time-photometric-stereo-and-surface-reconstruction-on-personal-devices/blob/master/example/sample_object.png)

- Initial estimate:

![alt text](https://github.com/hyc96/Real-time-photometric-stereo-and-surface-reconstruction-on-personal-devices/blob/master/example/estimate.png)

- Output:

![alt text](https://github.com/hyc96/Real-time-photometric-stereo-and-surface-reconstruction-on-personal-devices/blob/master/example/out.png)

## License

see the [LICENSE.md](LICENSE.md) file for details

## References 
This project is generally implemented based on:
1. Schindler, Photometric Stereo via Computer Screen Lighting for Real-time Surface Reconstruction. (2008)
2. H. Hideki, “Photometric stereo under a light-source with arbitrary motion”(1994)
3. A. Yuille, D. Snow, R. Epstein, and P. Belhumeur, “Determining generative models of objects under varying
illumination: Shape and albedo from multiple images using SVD and integrability”(1999)
4. V. Nozick, I. Daribo, and H. Saito, “Gpu-based photometric reconstruction from screen light” (2008)
5. V. Nozick, “Pyramidal Normal Map Integration for Real-time Photometric Stereo”(2010)

Contact me for more detailed report on the implementation.

## Contact
huaiyuc@seas.upenn.edu
