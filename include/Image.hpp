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

class PixelValueRGB;

class PixelValueHSL{
 public:
  PixelValueHSL();
  PixelValueHSL(double H, double S, double L);
  PixelValueHSL(const PixelValueRGB& in);
  PixelValueRGB toRGB() const;
  void setHSL(double H, double S, double L);
  double H_;
  double S_;
  double L_;
};

class PixelValueRGB{
 public:
  PixelValueRGB(){
    R_ = 0.0;
    G_ = 0.0;
    B_ = 0.0;
  };
  PixelValueRGB(double R, double G, double B){
    setRGB(R,G,B);
  };
  PixelValueRGB(const PixelValueHSL& in){
    *this = in.toRGB();
  };
  PixelValueHSL toHSL() const{
    PixelValueHSL out;
    const double vMM = std::max(R_,std::max(G_,B_));
    const double vM = std::min(R_,std::min(G_,B_));
    const double vC = vMM-vM;
    double vHt = 0;
    if(vC>0){
      if(vMM==R_){
        vHt = std::fmod((G_-B_)/vC,6.0);
      } else if(vMM==G_){
        vHt = (B_-R_)/vC + 2.0;
      } else {
        vHt = (R_-G_)/vC + 4.0;
      }
    }
    out.H_ = vHt*60;
    out.L_ = (vMM+vM)/2;
    if(out.L_ > 0 && out.L_ < 1.0){
      out.S_ = vC/(1.0 - std::fabs(2*out.L_-1.0));
    } else {
      out.S_ = 0.0;
    }
    return out;
  }
  void setRGB(double R, double G, double B){
    R_ = R;
    G_ = G;
    B_ = B;
  };
  void increaseL(double dL){
    PixelValueHSL HSL(*this);
    HSL.L_ += dL;
    if(HSL.L_ > 1.0) HSL.L_ = 1.0;
    if(HSL.L_ < 0.0) HSL.L_ = 0.0;
    *this = HSL.toRGB();
  }
  void setH(double H){
    PixelValueHSL HSL(*this);
    HSL.H_ = H;
    *this = HSL.toRGB();
  }
  void increaseH(double dH){
    PixelValueHSL HSL(*this);
    HSL.H_ = std::fmod(HSL.H_+dH+360,360);
    *this = HSL.toRGB();
  }
  double R_;
  double G_;
  double B_;
};

PixelValueHSL::PixelValueHSL(){
  H_ = 0.0;
  S_ = 0.0;
  L_ = 0.0;
};
PixelValueHSL::PixelValueHSL(double H, double S, double L){
  setHSL(H,S,L);
};
PixelValueHSL::PixelValueHSL(const PixelValueRGB& in){
  *this = in.toHSL();
};
void PixelValueHSL::setHSL(double H, double S, double L){
  H_ = H;
  S_ = S;
  L_ = L;
};
PixelValueRGB PixelValueHSL::toRGB() const{
  PixelValueRGB out;
  const double vC = (1.0-std::fabs(2*L_-1.0))*S_;
  const double vHt = std::fmod(H_,360.0)/60;
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
  const double vM = L_-0.5*vC;
  out.R_ = vR1+vM;
  out.G_ = vG1+vM;
  out.B_ = vB1+vM;
  return out;
};

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
  PixelValueRGB data_[W][H];

  Image(){
    reset();
  };
  void reset(){
    for(unsigned int i=0;i<height_;i++){
      for(unsigned int j=0;j<width_;j++){
        data_[i][j].setRGB(0.5,0.5,0.5);
      }
    }
  }
  void writeToFile(std::string fileName){
    FILE* fp = fopen(fileName.c_str(), "wb");

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
        temp[0] = (unsigned char)(data_[i][j].R_*255);
        temp[1] = (unsigned char)(data_[i][j].G_*255);
        temp[2] = (unsigned char)(data_[i][j].B_*255);
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
  PixelValueRGB& getPixelRGB(unsigned int i, unsigned int j){
    return data_[i][j];
  }
};

}

#endif /* IMAGE_HPP_ */
