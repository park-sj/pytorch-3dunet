import time

import numpy as np
import torch
import SimpleITK as sitk
import os
import skimage.transform

from pytorch3dunet.datasets.utils import SliceBuilder
from pytorch3dunet.unet3d.utils import get_logger
from pytorch3dunet.unet3d.utils import remove_halo
import pytorch3dunet.augment.transforms as transforms

logger = get_logger('UNet3DPredictor')


class _AbstractPredictor:
    def __init__(self, model, loader, output_file, config, **kwargs):
        self.model = model
        self.loader = loader
        self.output_file = output_file
        self.config = config
        self.predictor_config = kwargs

    @staticmethod
    def _volume_shape(dataset):
        # TODO: support multiple internal datasets
        raw = dataset.raws[0]
        if raw.ndim == 3:
            return raw.shape
        else:
            return raw.shape[1:]

    @staticmethod
    def _get_output_dataset_names(number_of_datasets, prefix='predictions'):
        if number_of_datasets == 1:
            return [prefix]
        else:
            return [f'{prefix}{i}' for i in range(number_of_datasets)]

    def predict(self):
        raise NotImplementedError
        
        
class _AbstractDicomPredictor(_AbstractPredictor):
    '''
    Dicom 파일로 출력을 저장하는 predictor를 위한 abstract predictor
    
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    @staticmethod
    def _save_dicom(newArray, transform, filepath, template_path):
        def _load_template(dir):
            logger.info("The template DCM directory is " + dir)
            assert os.path.isdir(dir), 'Cannot find the template directory'
            reader = sitk.ImageSeriesReader()
            dicomFiles = reader.GetGDCMSeriesFileNames(dir)
            reader.SetFileNames(dicomFiles)
            reader.MetaDataDictionaryArrayUpdateOn()
            reader.LoadPrivateTagsOn()
            image = reader.Execute()
            img3d = sitk.GetArrayFromImage(image)
            return image, reader, img3d.shape

        if not os.path.isdir(filepath):
            os.mkdir(filepath)

        oldImage, reader, imgShape = _load_template(template_path)
        logger.info("The template is loaded.")
        
        if len(newArray.shape) > 3:
            newArray = np.squeeze(newArray)
        newArray = skimage.transform.resize(newArray, imgShape, anti_aliasing=False)
        newArray = transform(newArray)
        newArray = newArray.astype(np.int8)
        
        newImage = sitk.GetImageFromArray(newArray)
        newImage = sitk.Cast(newImage, sitk.sitkInt8)
        # newImage.CopyInformation(oldImage)
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()
        sp_x, sp_y = reader.GetMetaData(0, "0028|0030").split('\\')
        # sp_z = reader.GetMetaData(0, "0018|0050")
        _, _, z_0 = reader.GetMetaData(0, "0020|0032").split('\\')
        _, _, z_1 = reader.GetMetaData(1, "0020|0032").split('\\')
        spacing_ratio = np.array([1, 1, 1], dtype=np.float64)
        sp_z = abs(float(z_0) - float(z_1))
        sp_z = float(sp_z) / spacing_ratio[0]
        sp_x = float(sp_x) / spacing_ratio[1]
        sp_y = float(sp_y) / spacing_ratio[2]
        modification_time = time.strftime("%H%M%S")
        modification_date = time.strftime("%Y%m%d")
        direction = newImage.GetDirection()
        series_tag_values = [(k, reader.GetMetaData(0, k)) for k in reader.GetMetaDataKeys(0)] + \
                             [("0008|0031", modification_time),
                             ("0008|0021", modification_date),
                             ("0028|0100", "8"),
                             ("0028|0101", "8"),
                             ("0028|0102", "7"),
                             ("0028|0103", "1"),
                             ("0028|0002", "1"),
                             ("0008|0008", "DERIVED\\SECONDARY"),
                             ("0020|000e", "1.2.826.0.1.3680043.2.1125." + modification_date + ".1" + modification_time),
                             ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6], direction[1], direction[4], direction[7]))))]
    #                         ("0008|103e", reader.GetMetaData(0, "0008|103e") + " Processed-SimpleITK")]
    #    print(series_tag_values)
        logger.info(f'Saving mask into {filepath}')
        tags_to_skip = ['0010|0010', '0028|0030', '7fe0|0010', '7fe0|0000', '0028|1052',
                        '0028|1053', '0028|1054', '0010|4000', '0008|1030', '0010|1001',
                        '0008|0080', '0010|0040', '0008|1010']
        for i in range(newImage.GetDepth()):
            image_slice = newImage[:, :, i]
            # image_slice.CopyInformation(oldImage[:, :, i])
            for tag, value in series_tag_values:
                if (tag in tags_to_skip):
                    continue
                if i == 0:
                    try:
                        logger.info(f'{tag} | {value}')
                    except:
                        continue
                image_slice.SetMetaData(tag, value)
            image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
            image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))
            image_slice.SetMetaData('0020|0032', reader.GetMetaData(i, "0020|0032"))
            image_slice.SetMetaData("0020|0013", str(i))
            image_slice.SetMetaData('0028|0030', '\\'.join(map(str, [sp_x, sp_y])))
            image_slice.SetSpacing([sp_x, sp_y])
            image_slice.SetMetaData("0018|0050", str(sp_z))
            writer.SetFileName(os.path.join(filepath, str(i).zfill(3) + '.dcm'))
            writer.Execute(image_slice)
        logger.info(f'Saved mask into {filepath}')


class StandardPredictor(_AbstractDicomPredictor):
    """
    dicom 영상 불러와서 통째로 forward 해서 저장하기 위한 predictor
    
    Args:
        model (Unet3D): trained 3D UNet model used for prediction
        data_loader (torch.utils.data.DataLoader): input data loader
        output_file (str): path to the output H5 file
        config (dict): global config dict
    """

    def __init__(self, model, loader, output_file, config, **kwargs):
        super().__init__(model, loader, output_file, config, **kwargs)
        self.it = 0
        self.file_paths = os.path.join(self.config['loaders']['test']['file_paths'][0])
        self.save_paths = os.path.join(self.config['loaders']['test']['save_paths'][0])

    def predict(self):
        out_channels = self.config['model'].get('out_channels')
        if out_channels is None:
            out_channels = self.config['model']['dt_out_channels']

        prediction_channel = self.config.get('prediction_channel', None)
        if prediction_channel is not None:
            logger.info(f"Using only channel '{prediction_channel}' from the network output")

        device = self.config['device']
        output_heads = self.config['model'].get('output_heads', 1)

        # stats are dummy values
        logger.info(f"Loading postprocessing... Configs {self.config['predictor']['transformer']}")
        transform = transforms.get_transformer(self.config['predictor']['transformer'], min_value=0, max_value=0,
                                                 mean=0, std=0)
        raw_transform = transform.raw_transform()
        
        logger.info(f'Running prediction on {len(self.loader)} batches...')
        
        # allocate prediction and normalization arrays
        logger.info('Allocating prediction and normalization arrays...')
        
        # Sets the module in evaluation mode explicitly (necessary for batchnorm/dropout layers if present)
        self.model.eval()
        # Set the `testing=true` flag otherwise the final Softmax/Sigmoid won't be applied!
        self.model.testing = True
        # Run predictions on the entire input dataset
        with torch.no_grad():
            for batch in self.loader:
                # send batch to device
                batch = batch.to(device)

                # forward pass
                predictions = self.model(batch)
                
                if output_heads == 1:
                    predictions = [predictions]

                for prediction in predictions:
                    prediction = prediction.cpu().numpy()
                    
                    
                    patients = os.listdir(self.file_paths)
                    logger.info(f"Predictions of {self.file_paths}")
                    logger.info(f"Saving predictions to: {self.save_paths}...")
                    self._save_dicom(prediction,
                                     raw_transform,
                                     os.path.join(self.save_paths, patients[self.it]),
                                     os.path.join(self.file_paths, patients[self.it]))
                    self.it += 1


class SkinPredictor(_AbstractDicomPredictor):
    """
    패치로 쪼개서 훈련된 모델을 위한 predictor

    Args:
        model (Unet3D): trained 3D UNet model used for prediction
        data_loader (torch.utils.data.DataLoader): input data loader
        output_file (str): path to the output H5 file
        config (dict): global config dict
    """

    def __init__(self, model, loader, output_file, config, **kwargs):
        super().__init__(model, loader, output_file, config, **kwargs)
        self.it = 0
        self.file_paths = os.path.join(self.config['loaders']['test']['file_paths'][0])
        self.save_paths = os.path.join(self.config['loaders']['test']['save_paths'][0])

    def predict(self):
        out_channels = self.config['model'].get('out_channels')
        if out_channels is None:
            out_channels = self.config['model']['dt_out_channels']

        prediction_channel = self.config.get('prediction_channel', None)
        if prediction_channel is not None:
            logger.info(f"Using only channel '{prediction_channel}' from the network output")

        device = self.config['device']
        output_heads = self.config['model'].get('output_heads', 1)

        # stats are dummy values
        logger.info(f"Loading postprocessing... Configs {self.config['predictor']['transformer']}")
        transform = transforms.get_transformer(self.config['predictor']['transformer'], min_value=0, max_value=0,
                                                 mean=0, std=0)
        raw_transform = transform.raw_transform()

        logger.info(f'Running prediction on {len(self.loader)} batches...')

        # dimensionality of the the output predictions
        volume_shape = tuple(self.loader.dataset.target_size)

        if prediction_channel is None:
            prediction_maps_shape = (out_channels,) + volume_shape
        else:
            # single channel prediction map
            prediction_maps_shape = (1,) + volume_shape

        logger.info(f'The shape of the output prediction maps (CDHW): {prediction_maps_shape}')

        patch_halo = self.predictor_config.get('patch_halo', (0, 0, 0))
        self._validate_halo(patch_halo, self.config['loaders']['test']['slice_builder'])
        logger.info(f'Using patch_halo: {patch_halo}')

        np_output_file = np.zeros(volume_shape)
        # allocate prediction and normalization arrays
        logger.info('Allocating prediction and normalization arrays...')
        prediction_maps, normalization_masks = self._allocate_prediction_maps(prediction_maps_shape,
                                                                              output_heads, np_output_file)

        # Sets the module in evaluation mode explicitly (necessary for batchnorm/dropout layers if present)
        self.model.eval()
        # Set the `testing=true` flag otherwise the final Softmax/Sigmoid won't be applied!
        self.model.testing = True
        # Run predictions on the entire input dataset
        with torch.no_grad():
            for batch, indices in self.loader:
                # send batch to device
                batch = batch.to(device)

                # forward pass
                predictions = self.model(batch)
                
                # wrap predictions into a list if there is only one output head from the network
                if output_heads == 1:
                    predictions = [predictions]

                # for each output head
                for prediction, prediction_map, normalization_mask in zip(predictions, prediction_maps,
                                                                          normalization_masks):

                    # convert to numpy array
                    prediction = prediction.cpu().numpy()

                    # for each batch sample
                    for pred, index in zip(prediction, indices):
                        # save patch index: (C,D,H,W)
                        if prediction_channel is None:
                            channel_slice = slice(0, out_channels)
                        else:
                            channel_slice = slice(0, 1)
                        # print(index)
                        index = (channel_slice,) + index

                        if prediction_channel is not None:
                            # use only the 'prediction_channel'
                            logger.info(f"Using channel '{prediction_channel}'...")
                            pred = np.expand_dims(pred[prediction_channel], axis=0)

                        logger.info(f'Saving predictions for slice:{index}...')

                        # remove halo in order to avoid block artifacts in the output probability maps
                        # u_prediction, u_index = remove_halo(pred, index, volume_shape, patch_halo)
                        u_prediction = pred
                        u_index = index
                        # accumulate probabilities into the output prediction array
                        prediction_map[u_index] += u_prediction
                        # count voxel visits for normalization
                        normalization_mask[u_index] += 1

        # save results to
        self._save_results(prediction_maps, normalization_masks, output_heads, np_output_file, self.loader.dataset, raw_transform)

    def _allocate_prediction_maps(self, output_shape, output_heads, output_file):
        # initialize the output prediction arrays
        prediction_maps = [np.zeros(output_shape, dtype='float32') for _ in range(output_heads)]
        # initialize normalization mask in order to average out probabilities of overlapping patches
        normalization_masks = [np.zeros(output_shape, dtype='uint8') for _ in range(output_heads)]
        return prediction_maps, normalization_masks

    def _save_results(self, prediction_maps, normalization_masks, output_heads, output_file, dataset, raw_transform):
        def _slice_from_pad(pad):
            if isinstance(pad, tuple):
                return slice(pad[0], -pad[1])
            elif pad == 0:
                return slice(None, None)
            else:
                return slice(pad, -pad)

        # save probability maps
        prediction_datasets = self._get_output_dataset_names(output_heads, prefix='predictions')
        for prediction_map, normalization_mask, prediction_dataset in zip(prediction_maps, normalization_masks,
                                                                          prediction_datasets):
            prediction_map = prediction_map / normalization_mask
            
            import math
            slice_builder_config = self.config['loaders']['test']['slice_builder']
            patch_shape = slice_builder_config['patch_shape']
            stride_shape = slice_builder_config['stride_shape']
            input_shape = self.loader.dataset.input_shape
            target_size = list(input_shape)
            for i in range(len(input_shape)):
                if input_shape[i] < patch_shape[i]:
                    target_size[i] = patch_shape[i]
                else:
                    n = math.ceil((input_shape[i]-patch_shape[i])/stride_shape[i])
                    target_size[i] = patch_shape[i] + n*stride_shape[i]
            padding_shape = ((math.ceil((target_size[0]-input_shape[0])/2), (math.floor((target_size[0]-input_shape[0])/2))),
                             (math.ceil((target_size[1]-input_shape[1])/2), (math.floor((target_size[1]-input_shape[1])/2))),
                             (math.ceil((target_size[2]-input_shape[2])/2), (math.floor((target_size[2]-input_shape[2])/2))))
            
            z_s, y_s, x_s = [_slice_from_pad(p) for p in padding_shape]
            logger.info(f'Padding shape : {padding_shape}, {z_s}, {y_s}, {x_s}')
            logger.info(f'Dataset loaded with mirror padding: {dataset.mirror_padding}. Cropping before saving...')

            prediction_map = prediction_map[:, z_s, y_s, x_s]
            logger.info(f"Prediction map shape : {prediction_map.shape}")

            logger.info(f'Saving predictions to: {self.output_file}/{prediction_dataset}...')
        logger.info(f"Max: {np.max(prediction_map)}, Min: {np.min(prediction_map)}")
        
        patients = os.listdir(self.file_paths)
        logger.info(f"Predictions of {self.file_paths}")
        logger.info(f"Saving predictions to: {self.save_paths}...")
        self._save_dicom(prediction_map,
                        raw_transform,
                        os.path.join(self.save_paths, patients[self.it]),
                        os.path.join(self.file_paths, patients[self.it]))
        self.it += 1

    @staticmethod
    def _validate_halo(patch_halo, slice_builder_config):
        patch = slice_builder_config['patch_shape']
        stride = slice_builder_config['stride_shape']

        patch_overlap = np.subtract(patch, stride)

        assert np.all(
            patch_overlap - patch_halo >= 0), f"Not enough patch overlap for stride: {stride} and halo: {patch_halo}"


class NpzPredictor(_AbstractPredictor):
    """
    Standard predictor which saves the prediction in npz format

    Args:
        model (Unet3D): trained 3D UNet model used for prediction
        data_loader (torch.utils.data.DataLoader): input data loader
        output_file (str): path to the output H5 file
        config (dict): global config dict
    """

    def __init__(self, model, loader, output_file, config, **kwargs):
        super().__init__(model, loader, output_file, config, **kwargs)
        self.it = 0
        self.file_paths = os.path.join(self.config['loaders']['test']['file_paths'][0])
        self.save_paths = os.path.join(self.config['loaders']['test']['save_paths'][0])

    def predict(self):
        out_channels = self.config['model'].get('out_channels')
        if out_channels is None:
            out_channels = self.config['model']['dt_out_channels']

        prediction_channel = self.config.get('prediction_channel', None)
        if prediction_channel is not None:
            logger.info(f"Using only channel '{prediction_channel}' from the network output")

        device = self.config['device']
        output_heads = self.config['model'].get('output_heads', 1)
        
        # stats are dummy values
        logger.info(f"Loading postprocessing... Configs {self.config['predictor']['transformer']}")
        transform = transforms.get_transformer(self.config['predictor']['transformer'], min_value=0, max_value=0,
                                                 mean=0, std=0)
        raw_transform = transform.raw_transform()

        logger.info(f'Running prediction on {len(self.loader)} batches...')

        # Sets the module in evaluation mode explicitly (necessary for batchnorm/dropout layers if present)
        self.model.eval()
        # Set the `testing=true` flag otherwise the final Softmax/Sigmoid won't be applied!
        self.model.testing = True
        # Run predictions on the entire input dataset
        with torch.no_grad():
            for batch in self.loader:
                # send batch to device
                batch = batch.to(device)

                # forward pass
                predictions = self.model(batch)
                
                if output_heads == 1:
                    predictions = [predictions]

                for prediction in predictions:
                    prediction = prediction.cpu().numpy()
                    
                    # save results
                    patients = os.listdir(self.file_paths)
                    logger.info(f'Saving prediction to: {os.path.join(self.save_paths, patients[self.it])}')
                    self._save_npz(prediction, os.path.join(self.save_paths, patients[self.it]), raw_transform)
                    self.it += 1

    @staticmethod
    def _save_npz(newArray, filepath, raw_transform):
        newArray = np.squeeze(newArray)
        # newArray = skimage.transform.resize(newArray, imgShape, anti_aliasing=False)
        newArray = raw_transform(newArray)
        np.savez_compressed(filepath, mask=newArray)
        logger.info(f'Saved mask into {filepath}')


class ABPredictor(_AbstractDicomPredictor):
    """
    Airway, bone 범위 입력받아서 분할하는 predictor

    Args:
        model (Unet3D): trained 3D UNet model used for prediction
        data_loader (torch.utils.data.DataLoader): input data loader
        output_file (str): path to the output H5 file
        config (dict): global config dict
    """

    def __init__(self, model, loader, output_file, config, **kwargs):
        super().__init__(model, loader, output_file, config, **kwargs)
        self.it = 0
        self.file_paths = os.path.join(self.config['loaders']['test']['file_paths'][0])
        self.save_paths = os.path.join(self.config['loaders']['test']['save_paths'][0])

    def predict(self):
        out_channels = self.config['model'].get('out_channels')
        if out_channels is None:
            out_channels = self.config['model']['dt_out_channels']

        prediction_channel = self.config.get('prediction_channel', None)
        if prediction_channel is not None:
            logger.info(f"Using only channel '{prediction_channel}' from the network output")

        device = self.config['device']

        logger.info(f'Running prediction on {len(self.loader)} batches...')


        # Sets the module in evaluation mode explicitly (necessary for batchnorm/dropout layers if present)
        self.model.eval()
        # Set the `testing=true` flag otherwise the final Softmax/Sigmoid won't be applied!
        self.model.testing = True
        # Run predictions on the entire input dataset
        with torch.no_grad():
            for batch in self.loader:
                if batch.shape[0] != 1:
                    raise NotImplementedError("Currently support running only on single data")
                
                logger.info(f'Image shpae is {batch.shape[2:]}')
                crop = input('Enter crop range in order with blank (X min, X max, Y min, Y max, Z min, Z max) :')
                crop = list(map(int, crop.split(' ')))
                
                assert crop[1] <= batch.shape[2] and \
                        crop[3] <= batch.shape[3] and \
                        crop[5] <= batch.shape[4], 'Entered crop range is out of image size'
                
                batch = batch[:,:,crop[0]:crop[1], crop[2]:crop[3], crop[4]:crop[5]]

                # send batch to device
                batch = torch.nn.functional.interpolate(batch, self.config['loaders']['test']['slice_builder']['patch_shape'])
                batch = batch.to(device)

                # forward pass
                prediction = self.model(batch)

                # convert to numpy array
                prediction = prediction.cpu().numpy()
                
                # save results to
                patient = os.listdir(self.file_paths)[0]
                self._save_dicom(prediction,
                                 os.path.join(self.file_paths, patient),
                                 os.path.join(self.save_paths, patient), crop)

