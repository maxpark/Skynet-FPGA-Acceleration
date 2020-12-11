#include "SkyNet.h"
void load_fm(ADT* fm, layer l)
{
    char nstr[50];
    sprintf(nstr, "./blob/%s.bb", l.name);
    FILE *fp = fopen(nstr, "rb");
    fread(fm, 1, l.ow*l.oh*l.oc * sizeof(ADT), fp);
    fclose(fp);
}

void load_weight(WDT32* weight , int length)
{
    char nstr[50];
    sprintf(nstr, "./weight/SkyNetT.wt");
    FILE *fp = fopen(nstr, "rb");
    fread(weight, 1, length*sizeof(WDT), fp);
    fclose(fp);
}
void load_biasm(BDT8* biasm , int length)
{
    char nstr[50];
    sprintf(nstr, "./weight/SkyNetT.bm");
    FILE *fp = fopen(nstr, "rb");
    fread(biasm, 1, length*sizeof(BDT8), fp);
    fclose(fp);
}

void check_fm(ADT* fm, layer l)
{
    int len = l.oc*l.ow*l.oh;
    ADT *tmp = (ADT *)malloc(sizeof(ADT)*len);

    char nstr[50];
    sprintf(nstr, "./blob/%s.bb", l.name);
    FILE *fp = fopen(nstr, "rb");
    fread(tmp, 1, len*sizeof(ADT), fp);
    fclose(fp);

    int err = 0;
    int zero;
    for(int c=0; c<l.oc; c++)
    {
        for(int h=0; h<l.oh; h++)
        {
            for(int w=0; w<l.ow; w++)
            {
                int index = c*l.oh*l.ow + h*l.ow + w;
                if (fm[index]!=tmp[index])
                {
                    err++;
                    //printf("correct %d, error: %d, c %d, w %d, h %d,\n", (int)tmp[index], (int)fm[index], c, w, h);
                }
            }
        }
    }

    if (err > 0)
        printf("%s error cnt= %d\n", l.name, err);
    else
        printf("%s correct \n", l.name);

    free(tmp);
}
