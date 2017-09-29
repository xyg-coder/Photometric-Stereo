Photometric Stero
=====

The math base of "Photometric Stero" can be easily got from Internet. You can experiment with the code and the materials here. And be careful to change the path in the main function of this code.<br>
I write `PhotometricStero` class and `test` and do some change to the PFMAccess from Dr.Qin.<br>
Below is the record of my idea.<br>

## Calculate the Normal map and Albedo map
* I use the `PFMAccess` to access the data in the `.pbm` file. Notice that the destructor of PFMAccess class will destroy the data in the float pointer, which is very dangerous. Thus I prohibit the copy constructor of the class. So, the vector can only store its pointer. Also remember to free the PFMAccess later.<br>
* Then I use the float data to construct the opencv Mat `Mat (int rows, int cols, int type, void *data, size_t step=AUTO_STEP)`, it can use the float data from the PFMAccess directly.<br>
* Also remember that the copy constructor of Mat is like `std::shared_ptr`. So you don't need to fear that redundant data copy.<br>
* I choose one mask of metal phere to get the max position of the phere to get the light direction.<br>
* Here, the `void	cv::minMaxLoc (InputArray src, double *minVal, double *maxVal=0, Point *minLoc=0, Point *maxLoc=0, InputArray mask=noArray())` is very useful since I can directly set the mask as one parameter.<br>
* Here, to get the circle of metal phere, I use `cv::findContours` to get the contour of the sphere. In order to avoid the potential noise, I adopt the contours with most points. Then, I initialize the position of circle center with the average position of circle points, and radius with the average distance of some points to the initial center. And I use some points to iterate to get the precise circle center and radius.<br>
* Then I use the brightest point on the lambertian phere to get the light intensity.<br>
* With the light direction and intensity, I get the normal and albedo.<br>
* Because the high error of the dark point on the object. Dr.Qin recommends us to omit 10% darkest points in every image. So I use the quickSelect algorithm to every image to get the 10% threshold. And these dark points won't attend later calculation.<br>
* And since the calculation needs at least 3 images to get the normal direction and albedo. I count the invalid image for every pixel. And only if valid image of one pixel is more than 3, this pixel will be calculated. Otherwise its value will be zero.<br>
* Then I use the matrix operation to get normal and albedo. We should notice here that we should normalize every pixel value with the source intensity of image. Moreover, since most pixels on the object is valid for all images, the pseudo-inverse of the whole L is usually used. So, I calculate it before the iteration.<br>

### effect:
When I map the normal to picture, I use the method mentioned in the `./Assignment_1/Assignment_1.pdf`.

| image type |apple | elephant | pear |
| ---------------|-------- | ---------- | ------------|
| normal map |![Image failed](./resultImage/appleNormal.jpg "apple normal map") | ![Image failed](./resultImage/elephantNormal.jpg "elephant normal map") | ![Image failed](./resultImage/pearNormal.jpg "pear normal map")|
| albedo map | ![Image failed](./resultImage/appleAlbedo.jpg "apple albedo map") | ![Image failed](./resultImage/elephantAlbedo.jpg "elephant albedo map") | ![Image failed](./resultImage/pearAlbedo.jpg "pear albedo map")|
| normal with albedo map | ![Image failed](./resultImage/appleNormalWithAlbedo.jpg "apple normal with albedo map") | ![Image failed](./resultImage/elephantNormalWithAlbedo.jpg "elephant normal with albedo map") | ![Image failed](./resultImage/pearNormalWithAlbedo.jpg "pear normal with albedo map")|

## Calculate the depth of every pixel
* For these invalid point which doesn't have normal, I take one `3*3` window and assign the average normal direction to the invalid point.<br>
* With every normal direction (nx, ny, nz) of every pixel, we have `(nx, ny, nz).dot((z(x+1, y)) - (z(x, y))) = 0, (nx, ny, nz).dot(z(x, y+1) - z(x, y)) = 0`. Thus we have `number_of_points * 2` equations.<br>
* When I try to solve the equation `Pz = b`, the rank of P is `number_of_points - 1`. Thus, I set the depth of the point with median index to be zero and move it to constant matrix `b`.<br>
* The parameter matrix `P` is of size `number_of_points * 2, number_of_points - 1`. So we have to create sparse matrix to solve the linear equation. I use `Eigen` library to solve it.<br>
* In the equation, if `nz` of one point is near to 0, the equation will be unstable. Thus, I make one judgement: `when nz is smaller than one constant, than add one small number to it`. I don't know if it is mathematically right. If you have better idea, please contact me at `xinyuan.gui95@gmail.com`.<br>

### Depth effect
It's kind of hard to map the depth to 2d-image because the depth of the object varies a lot. I try to use linear function and sigmoid function to map.<br>


#### linear transform without judgement of nz
| apple | elephant | pear |
| ---------------|-------- | ---------- |
| ![Image failed](./resultImage/appleHeight.jpg "apple height map") | ![Image failed](./resultImage/elephantHeightMethod1.jpg "elephant depth map") | ![Image failed](./resultImage/pearHeight.jpg "pear depth map")|



#### sigmoid transform with and without judgement of nz
|elephant with judgement of nz | elephant without judgement|
|------------------------------|---------------------------|
| ![Image failed](./resultImage/elephantHeightMethod2NoJudgement.jpg "elephant depth map without judgement") | ![Image failed](./resultImage/elephantHeightMethod2Judgement.jpg "elephant depth map with judgement")|

From the picture, we can see that the judgement decreases the outlier to some extent.<br>

### How to use
* In order to use it, you should have correctrly installed the `openCV` and `Eigen` library.<br>
* The main class is in `PhotometricStero.h` and `PhotometricStero.cc`.<br>
* `PFMAccess` class is used to access `.pbm` file, but you should never copy that. Use its pointer is a good idea.<br>
* The `testPhotometricStero.cpp` contains the `main` function and some examples. You should be careful to change the path.<br>
* project guide and materials is in the `./Assignment_1/`


If you have any question, contact me at `xinyuan.gui95@gmail.com`
