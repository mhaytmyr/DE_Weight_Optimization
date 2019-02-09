
########################
## CONGIF FOR Patch CBCT ###
filter1 = 8;
filter2 = 16;
filter3 = 32;
batchSize = 64; #64
stepsPerEpoch = int(446/batchSize);
valSteps = 45/batchSize;
downKernel = 1; upKernel = 3;
downPool = 2; upPool = 2;
numEpochs = 50;

#learning rate decay parameter
lrDecayRate = 1/(stepsPerEpoch*numEpochs);

#modelName = "2x512x512_16C3"
#modelName = "2x1024x768_8C3_16C2"
#modelName = "2x1024x768_8C3_16C3"

#this model uses, air+log normalization
#modelName = "2x1024x768_8C1_BN_16C1_BN"

#following is same as above, but air+standardize
#modelName = "2x1024x768_Norm1_8C1_BN_16C1_BN"

#following is same as above, but air+img/img.max()
#modelName = "2x1024x768_Norm2_8C1_BN_16C1_BN"

#not air norm, (img-global_mean)/global_std
#modelName = "2x1024x768_Norm3_8C1_BN_16C1_BN"

#not air norm, (img-global_mean)/global_std
#modelName = "2x384x256_Norm3_8C1_BN_16C1_BN"

#not air norm, (img-global_mean)/global_std
modelName = "2x384x256_Norm3_8C1_BN_16C1_BN"


ROWSTART = 200;
COLSTART = 300;
MARGIN = 50;
#H,W,C = 768, 1024, 2
H,W,C = 256, 384, 2
########################

#key: index of image, value = [col1,col2,row1,row2]
GOOD_DATA = {131: [445, 945, 140, 635], 262: [303, 904, 80, 674], 44: [224, 716, 74, 515],
        266: [313, 916, 73, 660], 139: [131, 961, 107, 678], 273: [321, 891, 69, 665], 274: [310, 863, 82, 657],
        275: [311, 841, 85, 608], 276: [304, 847, 72, 644], 277: [320, 777, 74, 631], 278: [301, 821, 62, 647],
        279: [327, 799, 80, 638], 280: [278, 873, 74, 657], 281: [277, 931, 68, 620], 282: [282, 888, 62, 619],
        283: [307, 924, 89, 679], 312: [290, 819, 74, 664], 287: [297, 850, 85, 681], 288: [313, 851, 86, 628],
        161: [162, 559, 103, 653], 291: [298, 807, 58, 668], 39: [170, 740, 70, 604], 40: [172, 744, 241, 695],
        41: [172, 738, 95, 528], 42: [173, 733, 46, 584], 43: [158, 737, 60, 693], 172: [187, 964, 104, 649],
        45: [153, 709, 247, 641], 303: [251, 948, 78, 683], 309: [277, 821, 77, 650], 56: [168, 714, 173, 621],
        185: [206, 990, 95, 679], 314: [302, 688, 34, 616], 315: [289, 787, 24, 609], 316: [303, 691, 47, 642],
        313: [312, 645, 47, 652], 64: [130, 721, 102, 590], 323: [221, 684, 43, 705], 289: [308, 890, 86, 631],
        329: [192, 692, 38, 638], 75: [227, 735, 111, 613], 77: [374, 783, 131, 601], 335: [139, 554, 44, 637],
        208: [269, 1024, 100, 697], 120: [102, 872, 85, 681], 57: [159, 726, 92, 583], 344: [78, 711, 10, 676],
        217: [278, 1014, 115, 641], 93: [330, 814, 142, 649], 351: [28, 552, 27, 676], 353: [26, 558, 42, 647],
        114: [89, 861, 91, 669], 145: [146, 994, 95, 669], 103: [61, 837, 139, 694], 237: [303, 966, 103, 653],
        242: [294, 1021, 119, 657], 246: [291, 1014, 114, 662], 248: [299, 1004, 76, 559], 121: [105, 858, 80, 589],
        250: [300, 983, 78, 681], 299: [258, 901, 65, 656]};
