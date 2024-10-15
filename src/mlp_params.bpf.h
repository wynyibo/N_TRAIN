#ifndef PARAMS_BPF_H
#define PARAMS_BPF_H

#define FIX_POINT 16

#define N0 6
#define N1 32
static const int32_t layer_0_weight[192] = {459, 25462, -44400, -20077, -17088, -21160, -20031, -10655, 22635, -21580, 21948, 25038, -18887, -30202, 6016, 5015, 18691, 21793, -19013, 6024, -26292, -4532, -39024, -30241, 6161, 19200, -36322, 22677, -18095, -16299, -18356, 35435, -46589, 15301, 14883, -7199, -8268, -1394, -26344, 23734, -27284, -2820, -18390, 32621, 14958, -22699, 589, -5753, -5542, 7809, -33236, -20183, -18239, -34390, 1046, 22985, 14935, -12186, -12777, -30587, -26890, -20125, 34676, 5380, 17156, 19503, -17594, 22530, 17793, -18265, 2935, -11881, -9672, 14305, -46515, 25691, -1876, -29, -1771, 27272, -36986, -33574, -15602, -8339, -9750, -1284, -1969, -23664, -37322, -13863, -8760, 17291, 11308, 11621, -16395, 5344, 12616, 4515, -10783, -36598, -33387, 1668, 5082, 23963, 10281, -8110, 1095, 23670, -32368, -29636, 31860, -13595, 34852, 6148, -7038, 55, 20429, 3875, -23176, -12846, 19168, -33411, 25786, 3392, 23329, 9631, -14273, 10366, 17219, 35432, 128, 309, -31993, -29068, -1357, 1200, -6089, 25548, -27810, -1302, -7919, -21286, -7146, -10292, 9137, 17787, -26033, -37102, -9874, -36805, -7529, 317, -10439, -29227, 1751, -20879, 34854, 21929, 26209, -19388, 8159, 11165, 1703, 10567, -20932, -19192, -10275, -31542, 28825, 16778, 5188, 18904, 19179, 29076, 31304, -30202, -13956, 5292, 26396, 31082, -7262, 19457, 825, 11330, -25138, 3984, -20320, -303, -464, -36598, -31518, -1133};

#define N2 32
static const int32_t layer_1_weight[1024] = {-12770, 163, 2946, -4584, 309, -7787, 3479, 17308, -916, 6843, 1289, 17326, 7686, 8156, -23288, 9360, 23354, 18552, -3513, 3544, 28066, 984, -11114, -8508, 3806, -302, 13245, 4886, 3346, 15242, -3775, -18778, -8097, -7842, -3753, 8419, 2212, -5460, -7814, -2700, 2691, -8378, 2135, -8603, -8739, -3985, 656, 1325, -12195, -8435, -10993, 1746, -7622, -9346, -6194, -9191, -4286, 1461, 1745, 4053, -6078, -3758, -1842, 9467, 11588, -15794, -1145, 16616, 17535, 412, 16128, -4692, 6902, 6984, -16043, 8443, 13913, 18944, 13274, 5179, 1215, -4553, -10857, 8070, -413, 17108, -16463, 9485, 18780, 12271, -6278, 17084, -7006, -19439, 3113, 6502, -6871, 8565, 10172, 8880, -13087, -14892, -11986, 4902, -6876, -11390, 19971, -11394, -2628, 2678, 10421, -14622, -11487, -1008, 14634, 10719, 4017, 711, 2246, 14164, 6934, -5741, 3165, -14066, 10305, 5387, 4966, -1688, -5474, -1268, 1860, 7900, -2837, 420, -15764, -1265, -14389, 12700, -246, 3803, 741, 2165, -11280, 3449, 20020, 16782, 14800, 4554, 11238, -149, 90, -23952, 12358, -5669, 21287, -9781, 15842, 4612, 13914, -9186, -1050, -4651, 1661, -11819, 8042, 18649, -10083, 6550, -2575, 3043, -4718, 10032, 3801, 1069, -13925, 8088, 3965, 16670, -1229, -15062, 22571, -5059, -2712, -3015, 1227, -14351, 6769, 14172, 8674, 12438, -3291, -5832, -27349, 2002, 13031, -2179, -3360, -15913, 3558, 5623, -18134, 3494, 18191, -4293, -5605, -9452, 1534, -9788, 1031, -7086, 21500, 12368, 9813, -11314, 9063, 6391, -1218, 1558, 19711, -8178, 10350, 13606, -19015, 10625, 20525, -12513, -18878, 5466, 10216, -5385, 3064, -1125, 3141, 7387, -6853, -1034, 1742, 7537, 20192, -3034, 17234, -4778, -3921, 284, -8228, -8254, -17153, 344, 15921, 10460, -7530, 10453, -11133, -20954, 8532, 17163, -10323, 10006, 6739, -7044, 2517, 3442, -11260, 8678, -13420, 1447, 14475, 2254, -12587, -3028, 9112, 5891, 7321, -4006, 19372, 8617, 12986, -4900, 18494, 8482, -6950, 2848, 21003, -16740, 15228, 6659, 5545, 2564, -22968, 4455, -4407, -16609, -10508, -8198, -5366, -2693, -17500, -21233, 3844, 12675, 5688, -25221, 1629, -12910, 97, -2054, 4786, -13281, 20662, 8595, -14148, -5402, 14095, -5843, 12236, 1555, 8365, 18899, 4183, -10528, -18695, 13944, 22077, 3620, -2882, -16378, 7679, 6371, -1524, -11525, 7653, 7957, -27719, -18035, 3914, -8273, -25233, -5196, 3252, 3269, 1765, -9707, 20687, 11991, -23827, 13257, -11115, -16035, -8757, 8001, -20104, 3011, 16843, -17507, 2106, -420, 12542, 19932, -2233, 5201, 5952, 2590, -3776, 6579, 9884, 8302, 13229, 16014, 13631, -5410, -14761, 17791, -17717, 13519, -11003, 1284, 10423, 6277, -9526, 2083, -873, -6109, 16571, 11103, -628, -10719, -19052, 4228, 1, 3228, 16446, 13102, 14204, 14347, -1609, -6404, 21002, 14490, 17459, -1042, 11510, -28, -8331, 8187, -2405, 11881, -7501, -6387, 20495, -1372, -4212, 16498, -3912, -19657, 14256, 11564, 5671, -20016, -16929, 9527, -525, 13035, 17628, 9735, 10813, 863, -20482, 11796, 22524, 238, 6219, 14127, 10535, -1703, -10837, 9736, -8202, -399, -2553, 3063, 21353, 20366, -9116, 13119, -5448, -19105, 6674, 7189, 10088, -12408, -18633, 11795, -6288, 17755, -7520, 8932, 7788, 14322, -17782, 8138, 6157, 10727, 14395, 8031, 6352, -8950, -3900, 13344, -10470, -8127, -13247, 11761, 21927, 5857, -14746, 9144, -8717, 765, 17772, 14740, -12854, 25332, 12759, -4581, -8573, -17800, 10775, -2585, -3630, -28798, 9711, 2819, -17267, -17057, -12287, -24563, 5716, -3133, 10749, -558, 6275, -2960, 13692, 15174, -6975, 8040, 23908, -11845, 11062, 20298, -15851, 14313, -1149, -6755, 3133, -11974, 726, -6869, -5465, 5899, -4240, 6946, -3950, 1040, 4689, 2245, -18825, 16086, 19563, 1302, -196, 6543, 23806, 3621, 2502, -7332, 15552, -9252, 17926, 1946, 15078, 9791, -1831, -1049, 22822, -14981, -19839, 15649, 5000, 296, -16187, 5433, 16476, 10073, -8887, 15451, 10593, 2790, 2831, 8290, 11746, -1465, -8122, -8302, -14685, -14672, -10005, 4728, 6368, 10080, 946, 16807, -4589, -19015, 4929, 16375, 7998, -8100, -4432, -7214, 1787, -1399, -11230, 13046, -7894, -3213, 12015, 5197, -2286, 8442, -11394, 18213, 7712, 17118, 314, 7370, 5245, -2338, -7, -9474, -8484, -6672, 7252, 5927, 3741, -5673, 7979, -4881, 14350, -11706, -6989, 11353, 14721, 3679, 5761, 8820, 13101, 15337, -16004, 14766, 16806, 15505, 10524, 13293, 17131, -9223, -8085, 136, 231, -3644, 2963, 871, -1185, 4097, -7496, 8711, -1587, -11013, 3384, 14825, 9665, -19641, -2289, 5113, -6018, 17021, 1236, -1127, 23343, 18811, -22737, -5154, 13151, 19219, 20037, 12858, 13540, -340, -17139, -5207, -8095, -9582, -14932, 12515, 17863, 489, -9383, 21605, -4082, -11428, -2199, 15771, 6428, -2886, 5900, -4300, -8358, -1201, 3966, 4371, -8579, 5418, 10102, 1198, 4035, 5556, -2186, -12741, 1328, 1948, -11142, 4678, 11891, 2590, -2562, -9133, 6208, 5401, 16790, 5925, 11298, 17913, 6088, -13868, 13200, -3577, -16131, 831, 1728, 11405, 1676, 1180, 17269, 19656, -17463, -3985, 9262, 956, 11329, -4668, 16286, -2369, -14042, 8586, -11509, 15895, -7892, 10313, 15851, 10611, -13175, 25129, -5049, -18691, 14026, 6815, -12073, 3513, -850, -1790, -12109, -7153, 4116, -4286, -3799, -23718, -1402, 2639, 6619, -23924, 6581, -12885, 14425, 857, -6673, -8313, -1472, 5581, -8866, -1036, -598, -1283, 2489, 2588, -4857, 12234, -5291, -28091, 17218, -10138, 512, 16899, -3131, 4329, 13019, 15289, 18401, 14333, -1561, 8126, 2575, 6406, 20918, 13957, 14110, -5807, -16335, 6265, -14187, 3906, 4723, 2764, 10046, 6792, -16059, 21323, 3258, -2461, -4171, 6282, 229, 8468, -761, -11405, -3868, -2357, -9098, 12697, -13077, 11051, -2907, 14983, 5940, 8010, -5018, 1252, 6396, 18375, 13160, 2763, 15084, 6858, 171, -14653, -82, -4260, 14648, 5597, 14109, 18006, 12151, -18411, 10804, -11574, -13779, 18194, 10303, 5434, 1260, 1933, 10794, 9057, -13640, 1646, 17723, 13594, 15920, 19533, 19039, 1911, -13947, 2805, -17891, 366, -7749, 2304, 14001, 17467, -14326, 23288, -3057, -8330, 16064, -1164, 6664, -2045, 5947, 4631, 7291, 639, 12075, -3694, -13348, 6729, -7504, -4803, 2603, -10890, -11118, 4137, 4054, -1327, 8201, 499, 4304, 14043, 3492, 4343, -12268, -12561, -11463, 5305, 7818, 4477, 8819, -3440, 14183, -16487, -14610, 11102, 16129, -5316, -9632, 2184, 17012, 19427, -22015, 12619, 4451, 17397, 16730, 13983, 14502, 4312, -18041, 7632, -1470, -11557, 3488, 3411, 1809, 4729, 398, 5562, -17820, -9791, -3402, 4552, 17977, -20552, -16405, 5222, 13089, 18155, -2326, 2784, 20118, 7231, -6399, 7400, 13616, 6921, 13329, 16090, 4225, -14968, -18756, -2430, -18438, 13911, -1292, 6588, 14027, 12865, -2520, 12230, 2145, -5171, -2769, 3811, 12374, 121, -1736, 11776, 17529, 4509, 7716, 12836, 11008, 4416, -9266, 13654, 24008, 4489, -2645, 13711, -3219, -617, -11306, -464, -1102, -3435, -11735, -691, 12560, 5729, 4251, 18062, -15855, -13136, 9494, -413, 21721, -19914, -13151, 21189, 9585, 10128, 16978, 11316, 5592, 3912, -8950, -1079, 10172, 604, 12615, 6465, 12496, -12373, -18224, 8781, -19874, 5147, -481, -2128, 18350, 9079, -2861, 22734, -21677, -8380, -3524, -1889};

#define N3 2
static const int32_t layer_2_weight[64] = {17806, 6200, -4606, 18396, 19739, 19959, 24738, -28956, 21114, 12584, 14432, -12206, -15414, -13939, -189, 28883, 12285, -17993, 4316, -14926, -10530, 16646, -15951, 3959, -19866, 15409, -21501, -8056, -18155, -21191, -11843, -21920, -13767, -3330, 19312, -16940, -18388, -21544, -23343, 23619, -20194, -28322, -24456, 14787, 12835, 17621, 19982, -11777, -23146, 6738, -18829, 15400, 16838, -5699, 25869, -6290, 13168, -11661, 6426, -1203, 22358, 6909, 17906, 3626};

// mean
static const int64_t mean[6] = {572, 11161066668, 36, 496, 12145, 379};

// scale
static const int64_t scale[6] = {451, 24027934608, 54, 1672, 49932, 1560};


#endif
