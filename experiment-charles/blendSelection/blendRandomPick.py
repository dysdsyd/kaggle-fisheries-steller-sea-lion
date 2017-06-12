# -*- coding: utf-8 -*-

bad_train_ids = (
    # From MismatchedTrainImages.txt
    3,       # Region mismatch
    # 7,     # TrainDotted rotated 180 degrees. Hot patch in load_dotted_image()
    9,       # Region mismatch
    21,      # Region mismatch
    30,      # Exposure mismatch -- not fixable
    34,      # Exposure mismatch -- not fixable
    71,      # Region mismatch
    81,      # Region mismatch
    89,      # Region mismatch
    97,      # Region mismatch
    151,     # Region mismatch
    184,     # Exposure mismatch -- almost fixable
    # 215,   # TrainDotted rotated 180 degrees. Hot patch in load_dotted_image()
    234,     # Region mismatch
    242,     # Region mismatch
    268,     # Region mismatch
    290,     # Region mismatch
    311,     # Region mismatch
    # 331,   # TrainDotted rotated 180 degrees. Hot patch in load_dotted_image()
    # 344,   # TrainDotted rotated 180 degrees. Hot patch in load_dotted_image()
    380,     # Exposure mismatch -- not fixable
    384,     # Region mismatch
    # 406,   # Exposure mismatch -- fixed by find_coords()
    # 421,   # TrainDotted rotated 180 degrees. Hot patch in load_dotted_image()
    # 469,   # Exposure mismatch -- fixed by find_coords()
    # 475,   # Exposure mismatch -- fixed by find_coords()
    490,     # Region mismatch
    499,     # Region mismatch
    507,     # Region mismatch
    # 530,   # TrainDotted rotated. Hot patch in load_dotted_image()
    531,     # Exposure mismatch -- not fixable
    # 605,   # In MismatchedTrainImages, but appears to be O.K.
    # 607,   # Missing annotations on 2 adult males, added to missing_coords
    614,     # Exposure mismatch -- not fixable
    621,     # Exposure mismatch -- not fixable
    # 638,   # TrainDotted rotated. Hot patch in load_dotted_image()
    # 644,   # Exposure mismatch, but not enough to cause problems
    687,     # Region mismatch
    712,     # Exposure mismatch -- not fixable
    721,     # Region mismatch
    767,     # Region mismatch
    779,     # Region mismatch
    # 781,   # Exposure mismatch -- fixed by find_coords()
    # 794,   # Exposure mismatch -- fixed by find_coords()
    800,     # Region mismatch
    811,     # Region mismatch
    839,     # Region mismatch
    840,     # Exposure mismatch -- not fixable
    869,     # Region mismatch
    # 882,   # Exposure mismatch -- fixed by find_coords()
    # 901,   # Train image has (different) mask already, but not actually a problem
    903,     # Region mismatch
    905,     # Region mismatch
    909,     # Region mismatch
    913,     # Exposure mismatch -- not fixable
    927,     # Region mismatch
    946,     # Exposure mismatch -- not fixable

    # Additional anomalies
    857,     # Missing annotations on all sea lions (Kudos: @depthfirstsearch)
)

train_nb = 948
tids = list(range(0, train_nb))
tids = list(set(tids) - set(bad_train_ids))

import random
import numpy as np
from itertools import chain

random.shuffle(tids)
tids = np.array(tids)

blendingSet = set()
while len(blendingSet) < 250 :
    pick = np.random.choice(tids)
    blendingSet = set(chain(blendingSet,set([pick])))

print(len(blendingSet))


#513, 2, 516, 8, 525, 15, 532, 20, 536, 537, 24, 538, 25, 29, 544, 546, 547, 37, 40, 41, 553, 43, 552, 559, 48, 50, 563, 52, 562, 566, 568, 570, 573, 577, 579, 68, 583, 72, 74, 588, 589, 78, 80, 82, 598, 87, 600, 91, 606, 94, 608, 612, 618, 619, 620, 115, 629, 635, 637, 126, 638, 131, 132, 645, 133, 647, 141, 142, 146, 147, 660, 661, 150, 664, 668, 157, 158, 670, 160, 673, 162, 674, 676, 677, 169, 170, 685, 175, 688, 689, 690, 691, 694, 695, 187, 188, 189, 702, 190, 708, 196, 198, 200, 201, 204, 206, 722, 211, 212, 215, 216, 217, 219, 732, 733, 735, 736, 223, 227, 229, 744, 233, 748, 238, 239, 240, 243, 246, 247, 253, 256, 259, 773, 774, 264, 266, 269, 270, 782, 788, 280, 792, 283, 796, 799, 288, 803, 805, 806, 295, 294, 809, 299, 300, 813, 814, 816, 304, 818, 308, 821, 309, 313, 827, 318, 831, 320, 319, 323, 836, 838, 328, 841, 842, 843, 331, 333, 846, 848, 339, 852, 340, 342, 343, 856, 347, 859, 864, 865, 354, 355, 359, 361, 874, 873, 366, 880, 369, 370, 371, 372, 884, 374, 373, 890, 379, 892, 381, 894, 383, 891, 897, 386, 385, 900, 389, 393, 400, 916, 924, 414, 929, 418, 931, 936, 426, 427, 942, 943, 433, 434, 435, 438, 439, 440, 446, 447, 448, 449, 451, 456, 458, 460, 464, 472, 483, 484, 485, 497, 505


trainingSet = set(tids) - blendingSet
print(trainingSet)
print(len(trainingSet))























