
# @License: The MIT License (MIT)
# @Copyright: Lab BICI2. All Rights Reserved.
#pylint: disable=C0103

import openslide_util as opu

slide_path = '../TCGA/sample.svs'

crop_size = (10000, 10000)

region, level_0_size = opu.crop_slideimage(slide_path,
                             location=(0, 0),
                            level=2, size=crop_size)

print level_0_size

#resize to 25% of the original cropped size
downsample_size = tuple([int(0.01*x) for x in level_0_size])

region = region.resize(downsample_size)
region.show()
