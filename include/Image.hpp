/*
 * State.hpp
 *
 *  Created on: Feb 9, 2014
 *      Author: Bloeschm
 */

#ifndef IMAGE_HPP_
#define IMAGE_HPP_

#include <iostream>
#include <unordered_map>
#include "kindr/rotations/RotationEigen.hpp"
namespace rot = kindr::rotations::eigen_impl;

namespace LWF{

template<unsigned int W, unsigned int H>
class Image{
 public:
  //header size in bytes - should be 40
  const unsigned int headerSize_ = 40;
  //width and height of the image
  const int width_ = W;
  const int height_ = H;
  //number of colour planes used
  const unsigned short nrPlanes_ = 1;
  //number of bits per pixel  - only accepting 8 and 24
  const unsigned short nrBits_ = 24;
  //indicates wether the data is compressed or not
  const unsigned int compression_ = 0;
  //this is the image size in bytes
  const unsigned int imageSize_ = 3*W*H + ((3*W)%4)*H;
  //pixels per meter - not sure what it's used for
  const int xRes_ = 0;
  const int yRes_ = 0;
  //the number of colours used if dealing with palette info
  const unsigned int nrColours_ = 0;
  //number of important colours.
  const unsigned int importantColours_ = 0;
  //this is the pallete (if used). Note that the BMP's that will be written will not be using palette indexing.
  const unsigned char *pallete_ = NULL;

  //id should always be 'B''M'
  const unsigned short id_ = ('M' << 8) + 'B';
  //this is the size of the file.
  const unsigned int fileSize_ = 14 + 40 + imageSize_;
  //we won't be using these variables
  const unsigned short int reserved1_ = 0;
  const unsigned short int reserved2_ = 0;
  //this is where the image data can be found - relative to the start of the file.
  const unsigned int offset_ = 14 + 40;

  // Image data
  unsigned char data_[W][H][3];

  Image(){
    reset();
  };
  void reset(){
    for(unsigned int i=0;i<height_;i++){
      for(unsigned int j=0;j<width_;j++){
        setPixelRGB(i,j,0.5,0.5,0.5);
      }
    }
  }
  void writeToFile(){
    FILE* fp = fopen("test.bmp", "wb");

    fwrite(&id_,sizeof(id_),1,fp);
    fwrite(&fileSize_,sizeof(fileSize_),1,fp);
    fwrite(&reserved1_,sizeof(reserved1_),1,fp);
    fwrite(&reserved2_,sizeof(reserved2_),1,fp);
    fwrite(&offset_,sizeof(offset_),1,fp);


    fwrite(&headerSize_,sizeof(headerSize_),1,fp);
    fwrite(&width_,sizeof(width_),1,fp);
    fwrite(&height_,sizeof(height_),1,fp);
    fwrite(&nrPlanes_,sizeof(nrPlanes_),1,fp);
    fwrite(&nrBits_,sizeof(nrBits_),1,fp);
    fwrite(&compression_,sizeof(compression_),1,fp);
    fwrite(&imageSize_,sizeof(imageSize_),1,fp);
    fwrite(&xRes_,sizeof(xRes_),1,fp);
    fwrite(&yRes_,sizeof(yRes_),1,fp);
    fwrite(&nrColours_,sizeof(nrColours_),1,fp);
    fwrite(&importantColours_,sizeof(importantColours_),1,fp);

    unsigned char temp[3];
    //now write the image data...
    for (unsigned int i=0; i<height_;i++){
      for (unsigned int j=0; j<width_;j++){
        temp[0] = data_[i][j][0];
        temp[1] = data_[i][j][1];
        temp[2] = data_[i][j][2];
        fwrite(temp,sizeof(temp[0]),3,fp);
      }
      //and introduce the padding
      temp[0] = 0;
      //write the padding - again the complicated expression ...
      for (unsigned int j=0; j<(4-(3*width_)%4)%4;j++)
        fwrite(temp,sizeof(temp[0]),1,fp);
    }

    fclose(fp);
  }
  void setPixelRGB(unsigned int i, unsigned int j, double R, double G, double B){
    data_[i][j][0] = (unsigned char)(R*255);
    data_[i][j][1] = (unsigned char)(G*255);
    data_[i][j][2] = (unsigned char)(B*255);
  }
  void setPixelFromHSL(unsigned int i, unsigned int j, double vH, double vS, double vL){
    const double vC = (1.0-std::fabs(2*vL-1.0))*vS;
    const double vHt = std::fmod(vH,360.0)/60;
    const double vX = vC*(1.0-std::fabs(std::fmod(vHt,2.0)-1.0));
    double vR1 = 0.0;
    double vG1 = 0.0;
    double vB1 = 0.0;
    if(vHt<1.0){
      vR1 = vC;
      vG1 = vX;
    } else if(vHt<2.0){
      vG1 = vC;
      vR1 = vX;
    } else if(vHt<3.0){
      vG1 = vC;
      vB1 = vX;
    } else if(vHt<4.0){
      vB1 = vC;
      vG1 = vX;
    } else if(vHt<5.0){
      vB1 = vC;
      vR1 = vX;
    } else if(vHt<6.0){
      vR1 = vC;
      vB1 = vX;
    }
    const double vM = vL-0.5*vC;
    data_[i][j][0] = vR1+vM;
    data_[i][j][1] = vG1+vM;
    data_[i][j][2] = vB1+vM;
  }
  void setPixelR(unsigned int i, unsigned int j, double R){
    data_[i][j][0] = (unsigned char)(R*255);
  }
  void setPixelG(unsigned int i, unsigned int j, double G){
    data_[i][j][1] = (unsigned char)(G*255);
  }
  void setPixelB(unsigned int i, unsigned int j, double B){
    data_[i][j][2] = (unsigned char)(B*255);
  }
  void getPixelHSL(unsigned int i, unsigned int j, double& vH, double& vS, double& vL){
    const double vR = data_[i][j][0];
    const double vG = data_[i][j][1];
    const double vB = data_[i][j][2];
    const double vMM = std::max(vR,std::max(vG,vB));
    const double vM = std::min(vR,std::min(vG,vB));
    const double vC = vMM-vM;
    // TODO
  }
};

}

#endif /* IMAGE_HPP_ */
