"""
Author: Manish Sapkota
Date: 04/21/2017
"""

import openslide as op


# def saveSlideAsJPEG(slide_path,save_path,level=0):
#     # check if slide can be opened
#     try:
#         Slide = op.OpenSlide(slide_path)
#     except op.OpenSlideError:
#         print("Cannot find file '" + slide_path + "'")
#         return
#     except op.OpenSlideUnsupportedFormatError:
#         print("File format is not OpenSlide supported type")
#     return


def crop_slideimage(slide_path,
                    location=(0, 0),
                    level=0,
                    size=(2000, 2000)
                    ):
    """
    Returns the region from the given whole_slide
    -----------
    Parameters
    -----------
    slide_path : str
    path and filename of slide
    location : tuple(x,y)
    top-left co-ordinates of the image location
    level :  integer
    pyramid level for use with OpenSlide's 'read_region'
    size : tuple(width, height)
    size of the region to be cropped

    -------
    Returns
    -------
    region : PIL.im
    cropped region as PIL image
    """
    # check if slide can be opened
    try:
        slide = op.OpenSlide(slide_path)
    except op.OpenSlideError:
        print "Cannot find file '" + slide_path + "'"
        return
    except op.OpenSlideUnsupportedFormatError:
        print "File format is not OpenSlide supported type"
        return

    level_0_size = slide.dimensions

    region = slide.read_region(location, level, level_0_size)

    return region, level_0_size
