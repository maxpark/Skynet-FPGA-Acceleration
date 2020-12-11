#include "SkyNet.h"

static layer config[layer_count] = {
{ "conv0", 320,160,32, 320,160,32, 0,0,0 },  //conv0
{ "conv1", 320,160,32, 320,160,32, 3,1,1 },  //conv1
{ "conv2", 320,160,32, 320,160,64, 1,1,0 },  //conv2
{ "pool1", 320,160,64, 160,80,64,  2,2,0 },  //pool1
{ "conv3", 160,80,64,  160,80,64,  3,1,1 },  //conv3
{ "conv4", 160,80,64,  160,80,96,  1,1,0 },  //conv4
{ "pool2", 160,80,96,  80,40,96,   2,2,0 },  //pool2
{ "conv5", 80,40,96,   80,40,96,   3,1,1 },  //conv5
{ "conv6", 80,40,96,   80,40,192,  1,1,0 },  //conv6
{ "reorg", 80,40,192,  40,20,768,  2,2,0 },  //reorg
{ "pool3", 80,40,192,  40,20,192,  2,2,0 },  //pool3
{ "conv7", 40,20,192,  40,20,192,  3,1,1 },  //conv7
{ "conv8", 40,20,192,  40,20,384,  1,1,0 },  //conv8
{ "conv9", 40,20,384,  40,20,384,  3,1,1 },  //conv9
{ "conv10",40,20,384,  40,20,512,  1,1,0 },  //conv10
{ "cat",   40,20,192,  40,20,1280, 0,0,0 },  //concat
{ "conv11",40,20,1280, 40,20,1280, 3,1,1 },  //conv11
{ "conv12",40,20,1280, 40,20,96,   1,1,0 },  //conv12
{ "conv13",40,20,96,   40,20,32,   1,1,0 },  //conv13
};

ADT FM1[32][43][83];
ADT FM2[32][43][83];
ADT FM3[32][43][83];
SDT FM4[32][43][83];

WDT WBUF3x3[3][32][3][3];
WDT WBUF1x1[2][32][32];

BDT BBUF[3][32];
MDT MBUF[3][32];

void REORG(ADT32 *ifm, ADT IFM[32][43][83], ap_uint<6> Cx, ap_uint<3> Rx)
{
#pragma HLS ARRAY_PARTITION variable=IFM dim=1 complete
    ap_uint<20> ifm_index;
    for (ap_uint<7> h = 1; h <= 41; h++)
    {
        for (ap_uint<7> w = 1; w <= 81; w++)
        {
#pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE II=1
            if (h != 21 && w != 41)
            {
                ap_uint<2> bias_h = (h >= 22) + (!Rx[1]);
                ap_uint<2> bias_w = (w >= 42) + (!Rx[0]);
                ap_uint<10> h_ = 2 * h - bias_h;
                ap_uint<10> w_ = 2 * w - bias_w;
                ap_uint<20> ifm_index = Cx * 83 * 163 + h_ * 163 + w_;
                ADT32 DATA = ifm[ifm_index];
                for (ap_uint<7> c = 0; c < 32; c++)
                {
#pragma HLS UNROLL
                    IFM[c][h][w] = DATA.range(8*c+7, 8*c);
                }
            }
        }
    }
}

BDT clamp_SDT(DT x, SDT min, SDT max)
{
#pragma HLS INLINE
    DT y;
    if(x<min) y=min;
    else if(x>max) y=max;
    else y = x;
    return y;
}

void DWCONV3X3(ADT IFM[32][43][83], SDT OFM[32][43][83], WDT WBUF3x3[32][3][3])
{
#pragma HLS ARRAY_PARTITION variable=OFM dim=1 complete
#pragma HLS ARRAY_PARTITION variable=IFM dim=1 complete
#pragma HLS ARRAY_PARTITION variable=WBUF3x3 dim=1 complete
    SDT odata = 0;
    for(int i=0; i<3; i++){
		for(int j=0; j<3; j++){
			for(int h=1; h<42; h++){
				for(int w=1; w<82; w++){
#pragma HLS PIPELINE II=1
					for(int c=0; c<32; c++){
                        odata = OFM[c][h][w];
                        odata += IFM[c][h+i-1][w+j-1]*WBUF3x3[c][i][j];
                        OFM[c][h][w] = odata;
					}
				}
			}
		}
	}
}

struct PE {
	int value;
	int count;
	int dstx;
};
struct REG {
	int x;
	int y;
};
void PWCONV1X1(ADT IFM[32][43][83], SDT OFM[32][43][83], WDT WBUF1x1[32][32])
{
	int b_col = 8;
	int size = 8;
	int dim1 = 32*32/size + 2*size -1;

    int localC[32][8];
#pragma HLS ARRAY_PARTITION variable=localC complete dim=0
    int A_buffer[8][143];
#pragma HLS ARRAY_PARTITION variable=A_buffer complete dim=1
	int B_buffer[8][143];
#pragma HLS ARRAY_PARTITION variable=B_buffer complete dim=1

	struct PE PE[size][size];
#pragma HLS ARRAY_PARTITION variable=PE complete dim=0
	struct REG regD[size][size];
#pragma HLS ARRAY_PARTITION variable=regD complete dim=0
	struct REG regQ[size][size];
#pragma HLS ARRAY_PARTITION variable=regQ complete dim=0

	for(int i=0; i < 32/size; i++) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4
#pragma HLS PIPELINE
		for(int y = 0; y < size; y++){
#pragma HLS LOOP_TRIPCOUNT min=8 max=8
			for(int x = 0; x < 32; x++){
#pragma HLS LOOP_TRIPCOUNT min=32 max=32
				A_buffer[y][i*32+y+x] = WBUF1x1[i*(size)+y][x];
			}
		}
	}


    for(int h=1; h<42; h++) {
#pragma HLS LOOP_TRIPCOUNT min=41 max=41
		for (int slide = 0; slide < 10; slide++) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=10
			for(int w= slide * b_col + 1; w < (slide + 1) * b_col + 1; w++) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=8
				for(int i=0; i < 32/size; i++) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4
#pragma HLS PIPELINE
					for(int y = 0; y < size; y++){
#pragma HLS LOOP_TRIPCOUNT min=8 max=8
						for(int x = 0; x < 32; x++){
#pragma HLS LOOP_TRIPCOUNT min=32 max=32
							B_buffer[y][i*32+y+x] = IFM[x][h][y + slide*b_col + 1];
						}
					}
				}
			}
/**************************Systolic Array Begin***********************************/
			for(int k = 0; k < dim1; k++) {
#pragma HLS LOOP_TRIPCOUNT min=143 max=143
#pragma HLS PIPELINE
				for(int x = 0; x < size; x++){
#pragma HLS LOOP_TRIPCOUNT min=8 max=8
				#pragma HLS UNROLL
					for(int y = 0; y < size; y++) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=8
					#pragma HLS UNROLL
						if (x+y == k % 32) {
							PE[x][y].dstx = x + size * (k/32);
							PE[x][y].count = 0;
							PE[x][y].value = 0;
						}
						if (x == 0 && y == 0) {
							PE[x][y].value += A_buffer[0][k] * B_buffer[0][k];
							regQ[x][y].x = A_buffer[0][k];
							regQ[x][y].y = B_buffer[0][k];
							PE[x][y].count++;
							if(PE[x][y].count == 32) {
								localC[PE[x][y].dstx][y] = PE[x][y].value;
								PE[x][y].value = 0;
								PE[x][y].count = 0;
							}
						}
						else if(x != 0 && y == 0) {
							PE[x][y].value += A_buffer[x][k] * regD[x-1][0].y;
							regQ[x][y].x = A_buffer[x][k];
							regQ[x][y].y = regD[x-1][0].y;
							PE[x][y].count++;
							if(PE[x][y].count == 32) {
								localC[PE[x][y].dstx][y] = PE[x][y].value;
								PE[x][y].value = 0;
								PE[x][y].count = 0;
							}
						}
						else if(x == 0 && y != 0) {
							PE[x][y].value += regD[0][y-1].x * B_buffer[y][k];
							regQ[x][y].x = regD[0][y-1].x;
							regQ[x][y].y = B_buffer[y][k];
							PE[x][y].count++;
							if(PE[x][y].count == 32) {
								localC[PE[x][y].dstx][y] = PE[x][y].value;
								PE[x][y].value = 0;
								PE[x][y].count = 0;
							}
						}
						else {
							PE[x][y].value += regD[x][y-1].x * regD[x-1][y].y;
							regQ[x][y].x = regD[x][y-1].x;
							regQ[x][y].y = regD[x-1][y].y;
							PE[x][y].count++;
							if(PE[x][y].count == 32) {
								localC[PE[x][y].dstx][y] = PE[x][y].value;
								PE[x][y].value = 0;
								PE[x][y].count = 0;
							}
						}
					}
				}
				for(int x = 0; x < size; x++){
#pragma HLS LOOP_TRIPCOUNT min=8 max=8
				#pragma HLS UNROLL
					for(int y = 0; y < size; y++) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=8
					#pragma HLS UNROLL
						regD[x][y] = regQ[x][y];
					}
				}
			}
			for(int w= slide * b_col + 1; w < (slide + 1) * b_col + 1; w++) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=8
				int col = w - slide * b_col - 1;
#pragma HLS PIPELINE
				for (int tm=0; tm<32; tm++) {
#pragma HLS LOOP_TRIPCOUNT min=32 max=32
					OFM[tm][h][w] = clamp_SDT(localC[tm][col] + OFM[tm][h][w], smin, smax);
				}
			}
		}
		for (int tm = 0; tm <32; tm++) {
#pragma HLS LOOP_TRIPCOUNT min=32 max=32
				int odata = 0;
#pragma HLS PIPELINE
				for (int tn = 0; tn < 32; tn++) {
#pragma HLS LOOP_TRIPCOUNT min=32 max=32
					odata += WBUF1x1[tm][tn]*IFM[tn][h][81];
				}
				OFM[tm][h][81] = clamp_SDT(odata + OFM[tm][h][81], smin, smax);
			}
    }
}
ADT clamp_adt(DT x, ADT min, ADT max)
{
#pragma HLS INLINE
    ADT y;
    if(x<min) y=min;
    else if(x>max) y=max;
    else y = x;
    return y;
}

DT ReLU(DT x)
{
#pragma HLS INLINE
    DT y;
    if(x<0) y=0;
    else y=x;
    return y;
}

void ACTIVATION(SDT IFM[32][43][83], ADT OFM[32][43][83], BDT BBUF[32], MDT MBUF[32])
{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=BBUF dim=1 complete
#pragma HLS ARRAY_PARTITION variable=MBUF dim=1 complete
    DT qy;
	ADT y;
	for(int h=0; h<43; h++){
		for(int w=0; w<83; w++){
#pragma HLS PIPELINE
			for(int c=0; c<32; c++){
                IFM[c][h][w] = IFM[c][h][w] + BBUF[c];
                IFM[c][h][w] = ReLU(IFM[c][h][w]);
                qy = IFM[c][h][w]*MBUF[c];
                qy = qy >> nm;
                y = clamp_adt(qy, amin, amax);
                if(h==0|h==42|w==0|w==82)
                    OFM[c][h][w] = 0;
                else
                    OFM[c][h][w] = y;
			}
		}
	}
}

void Load_WBUF3x3(WDT32* weight, WDT WBUF3x3[32][3][3], int Mx)
{
    for(int m=0; m<3; m++)
    {
        for(int n=0; n<3; n++)
        {
#pragma HLS PIPELINE II=1
            WDT32 DATA;
            DATA = weight[Mx*9+m*3+n];
            for(int c=0; c<32; c++)
            {
                WBUF3x3[c][m][n] = DATA.range(8*c+7,8*c);
            }
        }
    }
}

void Load_WBUF1x1(WDT32* weight, WDT WBUF1x1[32][32], int Mx, int Nx, int ic)
{
    for(int n=0; n<32; n++)
    {
#pragma HLS PIPELINE II=1
        WDT32 DATA;
        DATA = weight[Mx*ic+Nx*32+n];
        for(int m=0; m<32; m++)
        {
            WBUF1x1[m][n] = DATA.range(8*m+7,8*m);
        }
    }
}

void Load_BBUF(BDT8 *bias, BDT BBUF[32], int Mx)
{
#pragma HLS ARRAY_PARTITION variable=BBUF dim=1 complete
    for (int i=0; i<4; i++)
    {
#pragma HLS PIPELINE II=1
        for (int c=0; c<8; c++)
        {
            #ifdef __AP_INT__
            BBUF[i*8+c] = bias[Mx*4+i].range(32*c+nb-1, 32*c);
            #else
            BBUF[i*8+c] = bias[Mx*4+i].range(32*c+31, 32*c);
            #endif
        }
    }
}

void Load_FM(ADT32* ifm, ADT IFM[32][43][83], int Hx, int Wx, int Cx, int ow, int oh)
{
    int tile = ow/80;
    int h_o, w_o;
    if(tile)
    {
        h_o = Hx*40 + Hx/tile;
        w_o = Wx*80 + Wx/tile;
    }
    else
    {
        h_o = 0;
        w_o = 0;
    }
        
    for (int h=0; h<42; h++)
    {
        for (int w=0; w<82; w++)
        {
#pragma HLS PIPELINE II=1
            int ifm_index = Cx*(oh*2+3)*(ow*2+3) + (h+h_o)*(ow*2+3) + (w+w_o);
            ADT32 DATA;
            DATA = ifm[ifm_index];
            for (int c=0; c<32; c++)
            {
                IFM[c][h][w] = DATA.range(8*c+7,8*c);
            }
        }
    }
}

void Export_CONV(ADT32* fm, ADT OFM[32][43][83], int Hx, int Wx, int Cx, int ow, int oh)
{
    int tile = ow/80;
    int h_o, w_o;
    if(tile)
    {
        h_o = Hx*40 + Hx/tile;
        w_o = Wx*80 + Wx/tile;
    }
    else
    {
        h_o = 0;
        w_o = 0;
    }
    for (int h=1; h<=40; h++)
    {
        for (int w=1; w<=80; w++)
        {
#pragma HLS PIPELINE II=1
            int fm_index = Cx*(oh*2+3)*(ow*2+3) + (h+h_o)*(ow*2+3) + (w+w_o);
            ADT32 DATA;
            for (int c=0; c<32; c++)
            {
                DATA.range(8*c+7,8*c) = OFM[c][h][w];
            }
            fm[fm_index] = DATA;
        }
    }
}

ADT MAX(ADT a, ADT b, ADT c, ADT d)
{
#pragma HLS INLINE
	ADT t1 = a > b ? a : b;
	ADT t2 = c > d ? c : d;
	return t1 > t2 ? t1 : t2;
}

void POOL(ADT32* fm, ADT IFM[32][43][83], int Hx, int Wx, int Cx, int ow, int oh)
{
#pragma HLS ARRAY_PARTITION variable=IFM dim=1 complete
    int tile = ow/40;
    int h_o = Hx*20 + Hx/tile;
    int w_o = Wx*40 + Wx/tile;
    for (int h=1; h<=20; h++)
    {
        for (int w=1; w<=40; w++)
        {
#pragma HLS PIPELINE II=1
            int fm_index = Cx*(oh*2+3)*(ow*2+3) + (h+h_o)*(ow*2+3) + (w+w_o);
            ADT32 DATA;
            for (int c=0; c<32; c++){
                DATA.range(8*c+7,8*c) = MAX(IFM[c][2*h-1][2*w-1],IFM[c][2*h-1][2*w],IFM[c][2*h][2*w-1],IFM[c][2*h][2*w]);
			}
            fm[fm_index] = DATA;
        }
    }
}

void Load_FM1(ADT32* ifm, ADT IFM[32][43][83], int Cx)
{
    for (int h=0; h<43; h++)
    {
        for (int w=0; w<83; w++)
        {
#pragma HLS PIPELINE II=1
            int ifm_index = Cx*43*83 + h*83 + w;
            ADT32 DATA;
            DATA = ifm[ifm_index];
            for (int c=0; c<32; c++)
            {
                IFM[c][h][w] = DATA.range(8*c+7,8*c);
            }
        }
    }
}

void Export_CONV1(ADT32* fm, ADT OFM[32][43][83], int Cx)
{
    for (int h=0; h<43; h++)
    {
#pragma HLS PIPELINE II=1
        for (int c=0; c<32; c++)
        {
            OFM[c][h][41] = 0;
        }
    }

    for (int w=0; w<83; w++)
    {
#pragma HLS PIPELINE II=1
        for (int c=0; c<32; c++)
        {
            OFM[c][21][w] = 0;
        }
    }

    for (int h=1; h<42; h++)
    {
        for (int w=1; w<82; w++)
        {
#pragma HLS PIPELINE II=1
            int ofm_index = Cx*43*83 + h*83 + w;
            ADT32 DATA;
            for (int c=0; c<32; c++)
            {
                DATA.range(8*c+7,8*c) = OFM[c][h][w];
            }
            fm[ofm_index] = DATA;
        }
    }
}

void CLR_FM(SDT FM[32][43][83])
{
    for(int h=0; h<43; h++)
    {
        for(int w=0; w<83; w++)
        {
#pragma HLS PIPELINE II=1
            for(int c=0; c<32; c++)
            {
                FM[c][h][w] = 0;
            }
        }
    }
}

void Export_BBOX(BDT8* bbox, BDT8 BBOX[4])
{
    for (int i=0; i<4; i++)
    {
#pragma HLS PIPELINE II=1
        bbox[i] = BBOX[i];
    }
}

void Compute_BBOX(SDT OFM[32][43][83], BDT MBUF[32], BDT8 BBOX[4])
{
    int H,W;
    DT conf[2];
    SDT max[2], h_max[2], w_max[2];
    SDT xs[4], ys[4], ws[4], hs[4], flag[4], x[4], y[4];

    for(int b=0; b<4; b++)
    {
        switch(b)
        {
            case 0: H=1; W=1; break;
            case 1: H=1; W=42; break;
            case 2: H=22; W=1; break;
            case 3: H=22; W=42; break;
        }
        max[0] = OFM[4][H][W];
        h_max[0] = H;
        w_max[0] = W;
        max[1] = OFM[9][H][W];
        h_max[1] = H;
        w_max[1] = W;

        for(int h=0; h<20; h++){
            for(int w=0; w<40; w++){
#pragma HLS PIPELINE II=1
                if(OFM[4][h+H][w+W]>max[0]){
                    max[0] = OFM[4][h+H][w+W];
                    h_max[0] = h+H;
                    w_max[0] = w+W;
                }
                if(OFM[9][h+H][w+W]>max[1]){
                    max[1] = OFM[9][h+H][w+W];
                    h_max[1] = h+H;
                    w_max[1] = w+W;
                }
            }
        }
        conf[0] = max[0]*MBUF[4];
        conf[1] = max[1]*MBUF[9];
        if(conf[1]>conf[0])
        {
            xs[b] = OFM[5][h_max[1]][w_max[1]];
            ys[b] = OFM[6][h_max[1]][w_max[1]];
            ws[b] = OFM[7][h_max[1]][w_max[1]];
            hs[b] = OFM[8][h_max[1]][w_max[1]];
            flag[b] = 1;
            x[b] = w_max[1]-W;
            y[b] = h_max[1]-H;
        }
        else
        {
            xs[b] = OFM[0][h_max[0]][w_max[0]];
            ys[b] = OFM[1][h_max[0]][w_max[0]];
            ws[b] = OFM[2][h_max[0]][w_max[0]];
            hs[b] = OFM[3][h_max[0]][w_max[0]];
            flag[b] = 0;
            x[b] = w_max[0]-W;
            y[b] = h_max[0]-H;
        }
        BBOX[b].range(31,0)    = xs[b];
        BBOX[b].range(63,32)   = ys[b];
        BBOX[b].range(95,64)   = ws[b];
        BBOX[b].range(127,96)  = hs[b];
        BBOX[b].range(159,128) = flag[b];
        BBOX[b].range(191,160) = x[b];
        BBOX[b].range(223,192) = y[b];
    }
}

void Load_IMG(ADT4* img, ADT IFM[32][43][83], int Hx, int Wx, int b)
{
    int h_o = Hx*40-1;
    int w_o = Wx*80-1;
    for (int h=0; h<42; h++)
    {
        for (int w=0; w<82; w++)
        {
#pragma HLS PIPELINE II=1
            ADT4 DATA = img[b*320*160 + (h+h_o)*320 + (w+w_o)];
            for (int c=0; c<3; c++)
            {
                if (h+h_o<0||w+w_o<0||h+h_o>159||w+w_o>319)
                    IFM[c][h][w] = 128;
                else
                    IFM[c][h][w] = DATA.range(8*c+7,8*c);
            }
        }
    }
}

void SkyNet(ADT4* img, ADT32* fm, WDT32* weight, BDT8* biasm)
{
#pragma HLS INTERFACE m_axi depth=204800  port=img    offset=slave bundle=img
#pragma HLS INTERFACE m_axi depth=1514427 port=fm     offset=slave bundle=fm
#pragma HLS INTERFACE m_axi depth=13792   port=weight offset=slave bundle=wt
#pragma HLS INTERFACE m_axi depth=860     port=biasm  offset=slave bundle=bm
#pragma HLS INTERFACE s_axilite register  port=return

#pragma HLS ALLOCATION instances=Load_IMG		limit=1 function
#pragma HLS ALLOCATION instances=PWCONV1x1		limit=1 function
#pragma HLS ALLOCATION instances=DWCONV3x3	   	limit=1 function
#pragma HLS ALLOCATION instances=REORG	    	limit=1 function
#pragma HLS ALLOCATION instances=POOL	    	limit=1 function
#pragma HLS ALLOCATION instances=ACTIVATION    	limit=1 function
#pragma HLS ALLOCATION instances=Load_FM    	limit=1 function
#pragma HLS ALLOCATION instances=Export_CONV    limit=1 function
#pragma HLS ALLOCATION instances=Load_FM1    	limit=1 function
#pragma HLS ALLOCATION instances=Export_CONV1   limit=1 function

#pragma HLS ARRAY_PARTITION variable=WBUF1x1 complete dim = 1
#pragma HLS ARRAY_PARTITION variable=FM1 complete dim = 1
#pragma HLS ARRAY_PARTITION variable=FM2 complete dim = 1
#pragma HLS ARRAY_PARTITION variable=FM3 complete dim = 1
#pragma HLS ARRAY_PARTITION variable=FM4 complete dim = 1
    /*********************************DWCONV1+PWCONV1********************************/
    //std::cout << "DWCONV1+PWCONV1" << std::endl;
    Load_WBUF3x3(weight + conv1_w, WBUF3x3[0], 0);
    Load_BBUF(biasm + conv1_b, BBUF[0], 0);
    Load_BBUF(biasm + conv1_m, MBUF[0], 0);

    Load_WBUF1x1(weight + conv2_w, WBUF1x1[0], 0, 0, config[2].ic);
    Load_WBUF1x1(weight + conv2_w, WBUF1x1[1], 1, 0, config[2].ic);
    Load_BBUF(biasm + conv2_b, BBUF[1], 0);
    Load_BBUF(biasm + conv2_b, BBUF[2], 1);
    Load_BBUF(biasm + conv2_m, MBUF[1], 0);
    Load_BBUF(biasm + conv2_m, MBUF[2], 1);

    {
        for(int b=0; b<4; b++)
        {
            int H, W;
            switch(b)
            {
                case 0: H=0; W=0; break;
                case 1: H=0; W=4; break;
                case 2: H=4; W=0; break;
                case 3: H=4; W=4; break;
            }
            for(int Hx=0; Hx<4; Hx++)
            {
                Load_IMG(img, FM1, Hx, 0, b);
                for(int Wx=0; Wx<4; Wx++)
                {
                    if(Wx%2==0)
                    {
                        Load_IMG(img, FM2, Hx, Wx+1, b);
                        DWCONV3X3(FM1, FM4, WBUF3x3[0]);
                        ACTIVATION(FM4, FM1, BBUF[0], MBUF[0]);
                        CLR_FM(FM4);
                        Export_CONV(fm + conv1_o, FM1, Hx+H, Wx+W, 0, config[1].ow, config[1].oh);
                        for(int Mx=0; Mx<2; Mx++)
                        {
                            PWCONV1X1(FM1, FM4, WBUF1x1[Mx]);
                            ACTIVATION(FM4, FM3, BBUF[1+Mx], MBUF[1+Mx]);
                            CLR_FM(FM4);
                            Export_CONV(fm + conv2_o, FM3, Hx+H, Wx+W, Mx, config[2].ow, config[2].oh);
                            POOL(fm + pool1_o, FM3, Hx+H, Wx+W, Mx, config[3].ow, config[3].oh);
                            CLR_FM(FM4);
                        }
                    }
                    else
                    {
                        Load_IMG(img, FM1, Hx, Wx+1, b);
                        DWCONV3X3(FM2, FM4, WBUF3x3[0]);
                        ACTIVATION(FM4, FM2, BBUF[0], MBUF[0]);
                        CLR_FM(FM4);
                        Export_CONV(fm + conv1_o, FM2, Hx+H, Wx+W, 0, config[1].ow, config[1].oh);
                        for(int Mx=0; Mx<2; Mx++)
                        {
                            PWCONV1X1(FM2, FM4, WBUF1x1[Mx]);
                            ACTIVATION(FM4, FM3, BBUF[1+Mx], MBUF[1+Mx]);
                            CLR_FM(FM4);
                            Export_CONV(fm + conv2_o, FM3, Hx+H, Wx+W, Mx, config[2].ow, config[2].oh);
                            POOL(fm + pool1_o, FM3, Hx+H, Wx+W, Mx, config[3].ow, config[3].oh);
                            CLR_FM(FM4);
                        }
                    }
                }
            }
        }
    }
    /*********************************DWCONV2+PWCONV2********************************/
    //std::cout << "DWCONV2+PWCONV2" << std::endl;
    Load_WBUF3x3(weight + conv3_w, WBUF3x3[0], 0);
    Load_WBUF3x3(weight + conv3_w, WBUF3x3[1], 1);
    Load_BBUF(biasm + conv3_b, BBUF[0], 0);
    Load_BBUF(biasm + conv3_b, BBUF[1], 1);
    Load_BBUF(biasm + conv3_m, MBUF[0], 0);
    Load_BBUF(biasm + conv3_m, MBUF[1], 1);
    {
        for(int Hx=0; Hx<4; Hx++)
        {
            for(int Wx=0; Wx<4; Wx++)
            {
                Load_FM(fm + pool1_o, FM1, Hx, Wx, 0, config[4].ow, config[4].oh);
                DWCONV3X3(FM1, FM4, WBUF3x3[0]);
                ACTIVATION(FM4, FM1, BBUF[0], MBUF[0]);
                Export_CONV(fm + conv3_o, FM1, Hx, Wx, 0, config[4].ow, config[4].oh);

                CLR_FM(FM4);
                Load_FM(fm + pool1_o, FM2, Hx, Wx, 1, config[4].ow, config[4].oh);
                DWCONV3X3(FM2, FM4, WBUF3x3[1]);
                ACTIVATION(FM4, FM2, BBUF[1], MBUF[1]);
                Export_CONV(fm + conv3_o, FM2, Hx, Wx, 1, config[4].ow, config[4].oh);
                CLR_FM(FM4);
                for(int Mx=0; Mx<3; Mx++)
                {
                    Load_WBUF1x1(weight + conv4_w, WBUF1x1[0], Mx, 0, config[5].ic);
                    PWCONV1X1(FM1, FM4, WBUF1x1[0]);
                    Load_WBUF1x1(weight + conv4_w, WBUF1x1[1], Mx, 1, config[5].ic);
                    PWCONV1X1(FM2, FM4, WBUF1x1[1]);

                    Load_BBUF(biasm + conv4_b, BBUF[2], Mx);
                    Load_BBUF(biasm + conv4_m, MBUF[2], Mx);
                    ACTIVATION(FM4, FM3, BBUF[2], MBUF[2]);
                    Export_CONV(fm + conv4_o, FM3, Hx, Wx, Mx, config[5].ow, config[5].oh);
                    POOL(fm + pool2_o, FM3, Hx, Wx, Mx, config[6].ow, config[6].oh);
                    CLR_FM(FM4);
                }
            }
        }
    }
    /*********************************DWCONV3+PWCONV3********************************/
    //std::cout << "DWCONV3+PWCONV3" << std::endl;
    Load_WBUF3x3(weight + conv5_w, WBUF3x3[0], 0);
    Load_WBUF3x3(weight + conv5_w, WBUF3x3[1], 1);
    Load_WBUF3x3(weight + conv5_w, WBUF3x3[2], 2);
    Load_BBUF(biasm + conv5_b, BBUF[0], 0);
    Load_BBUF(biasm + conv5_b, BBUF[1], 1);
    Load_BBUF(biasm + conv5_b, BBUF[2], 2);
    Load_BBUF(biasm + conv5_m, MBUF[0], 0);
    Load_BBUF(biasm + conv5_m, MBUF[1], 1);
    Load_BBUF(biasm + conv5_m, MBUF[2], 2);
    {
        for(int Hx=0; Hx<2; Hx++)
        {
            for(int Wx=0; Wx<2; Wx++)
            {
                Load_FM(fm + pool2_o, FM1, Hx, Wx, 0, config[7].ow, config[7].oh);
                DWCONV3X3(FM1, FM4, WBUF3x3[0]);
                ACTIVATION(FM4, FM1, BBUF[0], MBUF[0]);
                CLR_FM(FM4);
                Export_CONV(fm + conv5_o, FM1, Hx, Wx, 0, config[7].ow, config[7].oh);

                Load_FM(fm + pool2_o, FM1, Hx, Wx, 1, config[7].ow, config[7].oh);
                DWCONV3X3(FM1, FM4, WBUF3x3[1]);
                ACTIVATION(FM4, FM1, BBUF[1], MBUF[1]);
                CLR_FM(FM4);
                Export_CONV(fm + conv5_o, FM1, Hx, Wx, 1, config[7].ow, config[7].oh);

                Load_FM(fm + pool2_o, FM1, Hx, Wx, 2, config[7].ow, config[7].oh);
                DWCONV3X3(FM1, FM4, WBUF3x3[2]);
                ACTIVATION(FM4, FM1, BBUF[2], MBUF[2]);
                CLR_FM(FM4);
                Export_CONV(fm + conv5_o, FM1, Hx, Wx, 2, config[7].ow, config[7].oh);
            }
        }
    }
    
    {
        for(int Hx=0; Hx<2; Hx++)
        {
            for(int Wx=0; Wx<2; Wx++)
            {
                for(int Mx=0; Mx<6; Mx++)
                {   
                    for(int Nx=0; Nx<3; Nx++)
                    {
                        Load_FM(fm + conv5_o, FM1, Hx, Wx, Nx, config[7].ow, config[7].oh);
                        Load_WBUF1x1(weight + conv6_w, WBUF1x1[1], Mx, Nx, config[8].ic);
                        PWCONV1X1(FM1, FM4, WBUF1x1[1]);
                    }
                    Load_BBUF(biasm + conv6_b, BBUF[0], Mx);
                    Load_BBUF(biasm + conv6_m, MBUF[0], Mx);
                    ACTIVATION(FM4, FM1, BBUF[0], MBUF[0]);
                    CLR_FM(FM4);
                    Export_CONV(fm + conv6_o, FM1, Hx, Wx, Mx, config[8].ow, config[8].oh);
                    POOL(fm + pool3_o, FM1, Hx, Wx, Mx, config[10].ow, config[10].oh);
                    CLR_FM(FM4);
                }
            }
        }
    }

    /*********************************DWCONV4+PWCONV4********************************/
    //std::cout << "DWCONV4+PWCONV4" << std::endl;
    {
        Load_FM1(fm + pool3_o, FM1, 0);
        for(int Nx=0; Nx<6; Nx++)
        {
            Load_WBUF3x3(weight + conv7_w, WBUF3x3[0], Nx);
            Load_BBUF(biasm + conv7_b, BBUF[0], Nx);
            Load_BBUF(biasm + conv7_m, MBUF[0], Nx);
            if(Nx%2==0)
            {
                Load_FM1(fm + pool3_o, FM2, Nx+1);
                DWCONV3X3(FM1, FM4, WBUF3x3[0]);
            }
            else
            {
                Load_FM1(fm + pool3_o, FM1, Nx+1);
                DWCONV3X3(FM2, FM4, WBUF3x3[0]);
            }
            ACTIVATION(FM4, FM3, BBUF[0], MBUF[0]);
            CLR_FM(FM4);
            Export_CONV1(fm + conv7_o, FM3, Nx);
        }
    }

    {
        for(int Mx=0; Mx<12; Mx++)
        {
            Load_BBUF(biasm + conv8_b, BBUF[0], Mx);
            Load_BBUF(biasm + conv8_m, MBUF[0], Mx);
            Load_FM1(fm + conv7_o, FM1, 0);
            for(int Nx=0; Nx<6; Nx++)
            {
                Load_WBUF1x1(weight + conv8_w, WBUF1x1[0], Mx, Nx, config[12].ic);
                if(Nx%2==0)
                {
                    Load_FM1(fm + conv7_o, FM2, Nx+1);
                    PWCONV1X1(FM1, FM4, WBUF1x1[0]);
                }
                else
                {
                    Load_FM1(fm + conv7_o, FM1, Nx+1);
                    PWCONV1X1(FM2, FM4, WBUF1x1[0]);
                }
            }
            ACTIVATION(FM4, FM2, BBUF[0], MBUF[0]);
            CLR_FM(FM4);
            Export_CONV1(fm + conv8_o, FM2, Mx);
        }
    }

    /*********************************DWCONV5+PWCONV5********************************/
    //std::cout << "DWCONV5+PWCONV5" << std::endl;
    {
        Load_FM1(fm + conv8_o, FM1, 0);
        for(int Nx=0; Nx<12; Nx++)
        {
            Load_WBUF3x3(weight + conv9_w, WBUF3x3[0], Nx);
            Load_BBUF(biasm + conv9_b, BBUF[0], Nx);
            Load_BBUF(biasm + conv9_m, MBUF[0], Nx);
            if(Nx%2==0)
            {
                Load_FM1(fm + conv8_o, FM2, Nx+1);
                DWCONV3X3(FM1, FM4, WBUF3x3[0]);
            }
            else
            {
                Load_FM1(fm + conv8_o, FM1, Nx+1);
                DWCONV3X3(FM2, FM4, WBUF3x3[0]);
            }
            ACTIVATION(FM4, FM3, BBUF[0], MBUF[0]);
            CLR_FM(FM4);
            Export_CONV1(fm + conv9_o, FM3, Nx);
        }
    }
    {
        Load_FM1(fm + conv9_o, FM1, 0);
        for(int Mx=0; Mx<16; Mx++)
        {
            Load_BBUF(biasm + conv10_b, BBUF[0], Mx);
            Load_BBUF(biasm + conv10_m, MBUF[0], Mx);
            Load_FM1(fm + conv9_o, FM1, 0);
            for(int Nx=0; Nx<12; Nx++)
            {
                Load_WBUF1x1(weight + conv10_w, WBUF1x1[0], Mx, Nx, config[14].ic);
                if(Nx%2==0)
                {
                    Load_FM1(fm + conv9_o, FM2, Nx+1);
                    PWCONV1X1(FM1, FM4, WBUF1x1[0]);
                }
                else
                {
                    Load_FM1(fm + conv9_o, FM1, Nx+1);
                    PWCONV1X1(FM2, FM4, WBUF1x1[0]);
                }
            }
            ACTIVATION(FM4, FM2, BBUF[0], MBUF[0]);
            CLR_FM(FM4);
            Export_CONV1(fm + conv10_o, FM2, Mx);
        }
    }
    /*********************************REORG+CONCAT+DWCONV6********************************/
    //std::cout << "REORG+DWCONV6" << std::endl;
    {
        for(int Nx=0; Nx<6; Nx++)
        {
            for(int Rx=0; Rx<4; Rx++)
            {
                REORG(fm + conv6_o, FM1, Nx, Rx);
                Load_WBUF3x3(weight + conv11_w, WBUF3x3[0], Nx+6*Rx);
                Load_BBUF(biasm + conv11_b, BBUF[0], Nx+6*Rx);
                Load_BBUF(biasm + conv11_m, MBUF[0], Nx+6*Rx);
                DWCONV3X3(FM1, FM4, WBUF3x3[0]);
                ACTIVATION(FM4, FM1, BBUF[0], MBUF[0]);
                CLR_FM(FM4);
                Export_CONV1(fm + conv11_o, FM1, Nx+6*Rx);
            }
        }
    }

    {
        Load_FM1(fm + conv10_o, FM1, 0);
        for(int Nx=0; Nx<16; Nx++)
        {
            Load_WBUF3x3(weight + conv11_w, WBUF3x3[0], Nx+24);
            Load_BBUF(biasm + conv11_b, BBUF[0], Nx+24);
            Load_BBUF(biasm + conv11_m, MBUF[0], Nx+24);
            if(Nx%2==0)
            {
                Load_FM1(fm + conv10_o, FM2, Nx+1);
                DWCONV3X3(FM1, FM4, WBUF3x3[0]);
            }
            else
            {
                Load_FM1(fm + conv10_o, FM1, Nx+1);
                DWCONV3X3(FM2, FM4, WBUF3x3[0]);
            }
            ACTIVATION(FM4, FM3, BBUF[0], MBUF[0]);
            CLR_FM(FM4);
            Export_CONV1(fm + conv11_o, FM3, Nx+24);
        }
    }
    /*********************************PWCONV6********************************/
    //std::cout << "PWCONV6" << std::endl;
    {
        for(int Mx=0; Mx<3; Mx++)
        {
            Load_FM1(fm + conv11_o, FM1, 0);
            for(int Nx=0; Nx<40; Nx++)
            {
                Load_WBUF1x1(weight + conv12_w, WBUF1x1[0], Mx, Nx, config[17].ic);
                if(Nx%2==0)
                {
                    Load_FM1(fm + conv11_o, FM2, Nx+1);
                    PWCONV1X1(FM1, FM4, WBUF1x1[0]);
                }
                else
                {
                    Load_FM1(fm + conv11_o, FM1, Nx+1);
                    PWCONV1X1(FM2, FM4, WBUF1x1[0]);
                }
            }
            Load_BBUF(biasm + conv12_b, BBUF[0], Mx);
            Load_BBUF(biasm + conv12_m, MBUF[0], Mx);
            ACTIVATION(FM4, FM3, BBUF[0], MBUF[0]);
            CLR_FM(FM4);
            Export_CONV1(fm + conv12_o, FM3, Mx);
        }
    }

    /*********************************CONV13********************************/
    //std::cout << "CONV13" << std::endl;
    for(int Nx=0; Nx<3; Nx++)
    {
        Load_FM1(fm + conv12_o, FM1, Nx);
        Load_WBUF1x1(weight + conv13_w, WBUF1x1[0], 0, Nx, config[18].ic);
        PWCONV1X1(FM1, FM4, WBUF1x1[0]);
    }
    Load_BBUF(biasm + conv13_m, MBUF[0], 0);
    BDT8 BBOX[4];
    Compute_BBOX(FM4, MBUF[0], BBOX);
    Export_BBOX(biasm + bbox_o, BBOX);
    CLR_FM(FM4);
}
