import numpy as np
#from ctapipe.calib.camera import integrators fix import with updated cta
from ctapipe.io import zfits
import logging,sys
from tqdm import tqdm
from utils.logger import TqdmToLogger
from utils.toy_reader import ToyReader
import matplotlib.pyplot as plt
# noinspection PyProtectedMember

def run(hist, options):
    # Few counters
    level, evt_num, first_evt, first_evt_num = 0, 0, True, 0

    log = logging.getLogger(sys.modules['__main__'].__name__+'.'+__name__)
    pbar = tqdm(total=len(options.scan_level)*options.events_per_level)
    tqdm_out = TqdmToLogger(log, level=logging.INFO)

    trig_out_mapping = ((204, 216, 229, 241, 254, 266, 279, 291, 304, 316, 329, 341, 180, 192, 205, 217, 230, 242, 255, 267, 280, 292, 305,
     317, 156, 168, 181, 193, 206, 218, 231, 243, 256, 268, 281, 293, 132, 144, 157, 169, 182, 194, 207, 219, 232, 244,
     257, 269, 108, 120, 133, 145, 158, 170, 183, 195, 208, 220, 233, 245, 84, 96, 109, 121, 134, 146, 159, 171, 184,
     196, 209, 221, 60, 72, 85, 97, 110, 122, 135, 147, 160, 172, 185, 197, 40, 50, 61, 73, 86, 98, 111, 123, 136, 148,
     161, 173, 24, 32, 41, 51, 62, 74, 87, 99, 112, 124, 137, 149, 12, 18, 25, 33, 42, 52, 63, 75, 88, 100, 113, 125, 4,
     8, 13, 19, 26, 34, 43, 53, 64, 76, 89, 101, 0, 2, 5, 9, 14, 20, 27, 35, 44, 54, 65, 77, 228, 239, 251, 262, 274,
     285, 297, 308, 320, 331, 343, 354, 240, 252, 263, 275, 286, 298, 309, 321, 332, 344, 355, 366, 253, 264, 276, 287,
     299, 310, 322, 333, 345, 356, 367, 377, 265, 277, 288, 300, 311, 323, 334, 346, 357, 368, 378, 387, 278, 289, 301,
     312, 324, 335, 347, 358, 369, 379, 388, 396, 290, 302, 313, 325, 336, 348, 359, 370, 380, 389, 397, 404, 303, 314,
     326, 337, 349, 360, 371, 381, 390, 398, 405, 411, 315, 327, 338, 350, 361, 372, 382, 391, 399, 406, 412, 417, 328,
     339, 351, 362, 373, 383, 392, 400, 407, 413, 418, 422, 340, 352, 363, 374, 384, 393, 401, 408, 414, 419, 423, 426,
     353, 364, 375, 385, 394, 402, 409, 415, 420, 424, 427, 429, 365, 376, 386, 395, 403, 410, 416, 421, 425, 428, 430,
     431, 215, 191, 167, 143, 119, 95, 71, 49, 31, 17, 7, 1, 227, 203, 179, 155, 131, 107, 83, 59, 39, 23, 11, 3, 238,
     214, 190, 166, 142, 118, 94, 70, 48, 30, 16, 6, 250, 226, 202, 178, 154, 130, 106, 82, 58, 38, 22, 10, 261, 237,
     213, 189, 165, 141, 117, 93, 69, 47, 29, 15, 273, 249, 225, 201, 177, 153, 129, 105, 81, 57, 37, 21, 284, 260, 236,
     212, 188, 164, 140, 116, 92, 68, 46, 28, 296, 272, 248, 224, 200, 176, 152, 128, 104, 80, 56, 36, 307, 283, 259,
     235, 211, 187, 163, 139, 115, 91, 67, 45, 319, 295, 271, 247, 223, 199, 175, 151, 127, 103, 79, 55, 330, 306, 282,
     258, 234, 210, 186, 162, 138, 114, 90, 66, 342, 318, 294, 270, 246, 222, 198, 174, 150, 126, 102, 78,),)

    trig_out_mapping_1 = ((132,299,133,311,120,134,323,298,121,135,335,310,108,122,136,347,322,297,109,123,137,359,334,
                           309,96,110,124,138,371,346,321,296,97,111,125,139,383,358,333,308,84,98,112,126,140,395,370,
                           345,320,295,85,99,113,127,141,407,382,357,332,307,72,86,100,114,128,142,419,394,369,344,319,
                           294,73,87,101,115,129,143,431,406,381,356,331,306,60,74,88,102,116,130,418,393,368,343,318,
                           293,61,75,89,103,117,131,430,405,380,355,330,305,48,62,76,90,104,118,417,392,367,342,317,
                           292,49,63,77,91,105,119,429,404,379,354,329,304,36,50,64,78,92,106,416,391,366,341,316,291,37
                           ,51,65,79,93,107,428,403,378,353,328,303,24,38,52,66,80,94,415,390,365,340,315,290,25,39,53,
                           67,81,95,427,402,377,352,327,302,12,26,40,54,68,82,414,389,364,339,314,289,13,27,41,55,69,83,
                           426,401,376,351,326,301,0,14,28,42,56,70,413,388,363,338,313,288,1,15,29,43,57,71,425,400,
                           375,350,325,300,144,2,16,30,44,58,412,387,362,337,312,145,156,3,17,31,45,59,424,399,374,349,
                           324,146,157,168,4,18,32,46,411,386,361,336,147,158,169,180,5,19,33,47,423,398,373,348,148,
                           159,170,181,192,6,20,34,410,385,360,149,160,171,182,193,204,7,21,35,422,397,372,150,161,172,
                           183,194,205,216,8,22,409,384,151,162,173,184,195,206,217,228,9,23,421,396,152,163,174,185,
                           196,207,218,229,240,10,408,153,164,175,186,197,208,219,230,241,252,11,420,154,165,176,187,
                           198,209,220,231,242,253,264,155,166,177,188,199,210,221,232,243,254,265,276,167,178,189,200,
                           211,222,233,244,255,266,277,179,190,201,212,223,234,245,256,267,278,191,202,213,224,235,246,
                           257,268,279,203,214,225,236,247,258,269,280,215,226,237,248,259,270,281,227,238,249,260,271,
                           282,239,250,261,272,283,251,262,273,284,263,274,285,275,286,287,),)
    batch = None
    for file in options.file_list:
        if level > len(options.scan_level) - 1:
            break
        # Get the file
        _url = options.directory + options.file_basename % file
        inputfile_reader = None
        if not options.mc:
            inputfile_reader = zfits.zfits_event_source(url=_url, max_events=len(options.scan_level)*options.events_per_level,expert_mode=True)
        else:

            seed = 0
            inputfile_reader = ToyReader(filename=_url, id_list=[0], seed=seed, max_events=len(options.scan_level)*options.events_per_level, n_pixel=options.n_pixels, events_per_level=options.events_per_level, level_start=options.scan_level[0])

        if options.verbose:
            log.debug('--|> Moving to file %s' % _url)
        # Loop over event in this file
        for event in inputfile_reader:
            if level > len(options.scan_level) - 1:
                break
            for telid in event.r0.tels_with_data:
                if first_evt:
                    first_evt_num = event.r0.tel[telid].camera_event_number
                    batch_index = 0
                    first_evt = False
                evt_num = event.r0.tel[telid].camera_event_number - first_evt_num
                if evt_num % options.events_per_level == 0:
                    level = int(evt_num / options.events_per_level)
                    if level > len(options.scan_level) - 1:
                        break
                    if options.verbose:
                        log.debug('--|> Moving to DAC Level %d' % (options.scan_level[level]))
                if options.events_per_level<=1000:
                    pbar.update(1)
                else:
                    if evt_num % int(options.events_per_level/1000)== 0:
                        pbar.update(int(options.events_per_level/1000))

                trig = np.array(list(event.r0.tel[telid].trigger_output_patch7.values()))
                trig = trig[trig_out_mapping_1]
                istrigged = np.any(trig>0.5,axis=-1)*np.ones((trig.shape[0],),dtype=int)
                hist.data[level,:]= hist.data[level,:]+istrigged
    # Update the errors
    hist._compute_errors()