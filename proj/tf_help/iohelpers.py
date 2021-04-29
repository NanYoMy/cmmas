# 0 - get configs
from sitkImageIO.itkdatareader import RegistorDataReader


def get_data_readers(dir_image0, dir_image1, dir_label0=None, dir_label1=None):

    reader_image0 = RegistorDataReader(dir_image0)
    reader_image1 = RegistorDataReader(dir_image1)

    reader_label0 = RegistorDataReader(dir_label0) if dir_label0 is not None else None
    reader_label1 = RegistorDataReader(dir_label1) if dir_label1 is not None else None

    # some checks
    # if not (reader_image0.num_data == reader_image1.num_data):
    #     raise Exception('Unequal num_data between images0 and images1!')
    # if dir_label0 is not None:
    #     if not (reader_image0.num_data == reader_label0.num_data):
    #         raise Exception('Unequal num_data between images0 and labels0!')
    #     if not (reader_image0.data_shape == reader_label0.data_shape):
    #         raise Exception('Unequal data_shape between images0 and labels0!')
    # if dir_label1 is not None:
    #     if not (reader_image1.num_data == reader_label1.num_data):
    #         raise Exception('Unequal num_data between images1 and labels1!')
    #     if not (reader_image1.data_shape == reader_label1.data_shape):
    #         raise Exception('Unequal data_shape between images1 and labels1!')
    #     if dir_label0 is not None:
    #         if not (reader_label0.num_labels == reader_label1.num_labels):
    #             raise Exception('Unequal num_labels between labels0 and labels1!')

    return reader_image0, reader_image1, reader_label0, reader_label1

