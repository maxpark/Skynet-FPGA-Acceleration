#ifndef SKYNET_H
#define SKYNET_H

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <memory.h>
#include <time.h>
#include <sys/time.h>
#include <fstream>
#include <cstring>
#include <math.h>
#include "ap_int.h"

#ifdef __SDSCC__
#include "sds_lib.h"
#else
#define sds_alloc malloc
#define sds_free free
#endif

#define na 8
#define nw 8
#define nb 19
#define nm 18
#define ns 21
#define qm 262144.0
typedef int DT;


//#define __AP_INT__
#ifdef __AP_INT__
typedef int DT;
typedef ap_uint<8> ADT;
typedef ap_int<nb> BDT;
typedef ap_int<8> WDT;
typedef ap_int<nb> MDT;
typedef ap_int<ns> SDT;
typedef ap_int<256> WDT32;
typedef ap_int<32>  ADT4;
typedef ap_int<256> ADT32;
typedef ap_int<1024> SDT32;
typedef ap_int<256> BDT8;
typedef ap_int<256> MDT8;
#else
typedef int DT;
typedef unsigned char ADT;
typedef int BDT;
typedef char WDT;
typedef int SDT;
typedef int MDT;
typedef ap_int<256> WDT32;
typedef ap_int<32>  ADT4;
typedef ap_int<256> ADT32;
typedef ap_int<1024> SDT32;
typedef ap_int<256> BDT8;
typedef ap_int<256> MDT8;
#endif


#define amin 0
#define amax 255
#define smin -1048576
#define smax 1048575


#define layer_count 19
#define check_scale 0.00001

struct layer
{
	char name[10];
	int iw, ih, ic, ow, oh, oc;
	int k, s, p;
};

#define pool1_o  0
#define pool2_o  105298
#define conv5_o  145885
#define conv6_o  186472
#define pool3_o  267646
#define conv7_o  289060
#define conv8_o  331888
#define conv9_o  374716
#define conv10_o 417544
#define conv11_o 474648
#define conv12_o 617408
#define conv1_o  628115
#define conv2_o  835804
#define conv3_o  1251182
#define conv4_o  1356480
#define fm_all   1514427

#define conv1_b 0
#define conv2_b 8
#define conv3_b 24
#define conv4_b 40
#define conv5_b 64
#define conv6_b 88
#define conv7_b 136
#define conv8_b 184
#define conv9_b 280
#define conv10_b 376
#define conv11_b 504
#define conv12_b 824
#define conv13_b 848

#define conv1_m 4
#define conv2_m 16
#define conv3_m 32
#define conv4_m 52
#define conv5_m 76
#define conv6_m 112
#define conv7_m 160
#define conv8_m 232
#define conv9_m 328
#define conv10_m 440
#define conv11_m 664
#define conv12_m 836
#define conv13_m 852
#define bbox_o 856

#define conv1_w 0
#define conv2_w 9
#define conv3_w 73
#define conv4_w 91
#define conv5_w 283
#define conv6_w 310
#define conv7_w 886
#define conv8_w 940
#define conv9_w 3244
#define conv10_w 3352
#define conv11_w 9496
#define conv12_w 9856
#define conv13_w 13696

/**********utils.cpp************/
void load_fm(ADT* fm, layer l);
void load_weight(WDT32 *weight, int length);
void load_biasm(BDT8* biasm , int length);
void check_fm(ADT* fm, layer l);


/**********transform.cpp************/
void stitch(ADT* ifm[4], ADT* ofm, layer l);
void distitch(ADT* ifm, ADT* ofm[4], layer l);
void img_DT_2_DT4(ADT* in, ADT4* out, layer l, int b);
void fm_DT_2_DT32(ADT* in, ADT32* out, layer l);
void fm_DT32_2_DT(ADT32* in, ADT* out, layer l);
void distitch_bbox(BDT* ifm, BDT* ofm[4], layer l);
/**********SkyNet.h [HW]************/
void SkyNet(ADT4* img, ADT32* fm, WDT32* weight, BDT8* biasm);

/**********operations************/
void pwconv1x1(DT *ifm, DT *ofm, DT *weight, DT *bias, int relu, layer l);
void dwconv3x3(DT *ifm, DT *ofm, DT *weight, DT *bias, int relu, layer l);
void maxpool(DT *ifm, DT *ofm, layer l);
void concat(DT *ifm1, DT *ifm2, DT *ofm, layer l1, layer l2);
void reorg(DT *ifm, DT *ofm, layer l);
#endif
