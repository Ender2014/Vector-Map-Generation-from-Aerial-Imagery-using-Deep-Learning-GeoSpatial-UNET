import os
import sys
from osgeo import gdal
import cv2
import numpy as np
import time as mtime
import argparse
import logging
import multiprocessing # <--- Import multiprocessing

# from matplotlib import pyplot as plt
from keras.models import load_model
# from keras.models import Model # Not explicitly used, can be commented out
# from keras import backend as K # Not explicitly used, can be commented out
import tensorflow as tf

from src import postprocess
from src import metric
from src import io
from src import util
from src import bf_grid
# from src import metric # Already imported above
from src import dataGenerator
from src import model

import config

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# _config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# --- Logger setup (can stay outside main) ---
util.check_dir(config.path_logs)
# Note: Corrected os.path.join usage
util.set_logger(os.path.join(config.path_logs, 'testing.log'))

# --- Define the main execution logic within a function ---
def main():
    parser = argparse.ArgumentParser(
        description='See description below to see all available options')

    parser.add_argument('-sg', '--skipGridding',
                        help='If skipping grididing while testing. [Default] False',
                        type=bool,
                        default=False,
                        required=False)

    parser.add_argument('-d', '--data',
                        help='Input Data folder where TIF files are stored',
                        type=str,
                        required=True)

    parser.add_argument('-ups', '--upscalling',
                        help='UpScale TIF image to predefined 10cm image',
                        type=bool,
                        required=True) # Consider default=False if it's optional

    parser.add_argument('-pt', '--pretrained',
                        help='Path of pretrained complete model or weight file.\
                             Use -w flag to mark it as weight or complete model',
                        type=str,
                        required=True)

    parser.add_argument('-w', '--weight',
                        help='If model provided is Model Weight or not. \
                            True - It is Weight, False- Complete Model',
                        type=bool,
                        required=True) # Consider default=False if it's usually a full model

    parser.add_argument('-lf', '--linearFeature',
                        help='If the object is linear feature like road? \
                            [Default] False',
                        type=bool,
                        default=False,
                        required=False)

    parser.add_argument('-o', '--output',
                        help='Output Data folder where TIF files will be saved',
                        type=str,
                        required=True)


    args = parser.parse_args()
    path_data = args.data
    st_time = mtime.time()

    logging.info('Input data given: {}'.format(path_data))
    logging.info('percent_overlap : {}'.format(config.overlap))

    # Storing time of process here
    timing = {}

    # Current running process
    logging.info('Initilization')

    # Filer for post processing
    filter = config.erosion_filter
    simplify_parameter = config.simplify_parameter  # in metres

    # Results path
    path_result = args.output
    path_tiled = os.path.join(path_result, 'tiled')
    path_predict = os.path.join(path_result, 'prediction')
    path_merged_prediction = os.path.join(path_result, 'merged_prediction')
    path_erosion = os.path.join(path_merged_prediction, 'erosion')
    path_watershed = os.path.join(path_merged_prediction, 'watershed')
    path_vector = os.path.join(path_merged_prediction, 'vector')
    path_simplify = os.path.join(path_merged_prediction, 'simplify')
    path_bbox = os.path.join(path_merged_prediction, 'bbox')

    # Creating directory
    util.check_dir(path_result)
    util.check_dir(path_predict)
    util.check_dir(path_tiled)
    util.check_dir(path_merged_prediction)
    util.check_dir(path_erosion)
    util.check_dir(path_watershed)
    util.check_dir(path_vector)
    util.check_dir(path_simplify)
    util.check_dir(path_bbox)

    # Logging output paths
    logging.info('Result path is %s' % (path_result))
    logging.info('Predict path is %s' % (path_predict))
    logging.info('Tile image path is %s' % (path_merged_prediction)) # Note: path_tiled is the input tiles, path_merged_prediction is output dir
    logging.info('Erosion path is %s' % (path_erosion))
    logging.info('watershed path is %s' % (path_watershed))
    logging.info('vector path is %s' % (path_vector))
    logging.info('simplify path is %s' % (path_simplify))
    logging.info('bbox path is %s' % (path_bbox))


    # loading model from model file or  weights file
    logging.info('Loading trained model')

    # --- Determine input shape based on config ---
    # It's good practice to define this once based on your config
    input_shape = (config.image_size, config.image_size, config.num_image_channels)

    if args.weight is True:
        # Pass the determined input shape to the model function
        unet_model = model.unet(config.image_size)
        try:
            unet_model.load_weights(args.pretrained)
            logging.info(f"Successfully loaded weights from {args.pretrained}")
            unet_model.summary(print_fn=logging.info) # Log model summary
        except Exception as e:
            msg = 'Unable to load model weights: {}'.format(args.pretrained)
            logging.error(msg)
            # Consider re-raising or exiting more gracefully
            raise Exception(f'{msg}. Error : {e}') # Use f-string and raise specific Exception

    else:
        try:
            # Ensure custom objects are correctly passed if needed by the saved model format
            unet_model = load_model(args.pretrained, custom_objects={
                'dice_coef': metric.dice_coef, 'jaccard_coef': metric.jaccard_coef}, compile=False) # Often safer to compile=False if you recompile later or just need inference
            # Optional: Re-compile if necessary for specific metrics/losses during potential evaluation,
            # but usually not needed just for predict. If you do, use an appropriate optimizer and loss.
            # unet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[metric.dice_coef, metric.jaccard_coef])
            logging.info(f"Successfully loaded complete model from {args.pretrained}")
            unet_model.summary(print_fn=logging.info) # Log model summary
        except Exception as e:
            msg = 'Unable to load model: {}'.format(args.pretrained)
            logging.error(msg)
            raise Exception(f'{msg}. Error : {e}') # Use f-string and raise specific Exception


    # Iterating over all the files
    for root, dirs, files in os.walk(path_data):
        # Filter files processed in this loop run to avoid reprocessing nested outputs
        files_to_process = [f for f in files if f.endswith(tuple(config.image_ext))]
        if not files_to_process:
             continue # Skip directories with no matching images

        logging.info(f"Found {len(files_to_process)} image(s) in {root}")

        for file in files_to_process:
            file_st_time = mtime.time() # Time each file separately
            print('Processing: {}'.format(file))
            logging.info(f"--- Processing file: {file} ---")
            temp_path_data = os.path.join(root, file)

            if args.upscalling is True:
                logging.info(f'Scaling TIF {file} to {config.default_resolution}m resolution')
                try:
                    # Converting to desired resolution data
                    res = config.default_resolution
                    # Use specific options for clarity and control
                    gdalOption = gdal.WarpOptions(
                            format='VRT', # VRT is efficient, doesn't duplicate data
                            xRes=res, yRes=res,
                            resampleAlg='cubic', # Choose a resampling alg (e.g., nearest, bilinear, cubic)
                            creationOptions=['TILED=YES'] # Optional: Tiled VRT
                        )

                    # Creating output file name
                    # Put VRT in the output 'tiled' dir to keep source clean? Or keep alongside source?
                    # Here keeping alongside source:
                    path_image_output = os.path.join(root, util.getNamenoExt(file) + '_' + str(res)+'.vrt')

                    # Creating VRT of input image
                    gdal.Warp(path_image_output, os.path.abspath(temp_path_data), options=gdalOption)
                    logging.info(f"Created scaled VRT: {path_image_output}")

                    # Replacing default image path with the path to the new VRT
                    temp_path_data = path_image_output

                except Exception as e:
                    logging.error(f"Failed to upscale {file}: {e}")
                    continue # Skip to next file if upscaling fails


            # Creating a new folder for tiled and prediction of the file
            _name = os.path.splitext(os.path.basename(file))[0]
            # Add resolution to distinguish if upscaled version is used
            if args.upscalling:
                _name += f"_{config.default_resolution}m"

            # Use unique name for tiled/predicted outputs based on the input file
            temp_path_tiled = os.path.join(path_tiled, _name) # Tiles specific to this input file
            temp_path_predict = os.path.join(path_predict, _name) # Predictions specific to this input file

            # Creating folders for this specific file's outputs
            util.check_dir(temp_path_tiled)
            util.check_dir(temp_path_predict)

            logging.info('Gridding image : {}'.format(os.path.basename(temp_path_data))) # Log basename for clarity

            if args.skipGridding is False:
                grid_time = mtime.time()
                try:
                    bf_grid.grid_file(path_data=temp_path_data,
                                      path_output=temp_path_tiled) # Use config
                    logging.info(f'Gridding completed for {os.path.basename(temp_path_data)} in {mtime.time() - grid_time:.2f}s')
                except Exception as e:
                    logging.error(f"Failed to grid {temp_path_data}: {e}")
                    continue # Skip to next file if gridding fails
            else:
                 logging.info(f"Skipping gridding for {os.path.basename(temp_path_data)}")


            # Loading the Testing image tiles
            logging.info('Reading Gridded image tiles from: {}'.format(temp_path_tiled))
            try:
                testing_dataList = dataGenerator.getTestingData(
                    path_tile_image=temp_path_tiled)

                testing_list_ids, testing_imageMap = testing_dataList.getList()
                if not testing_list_ids:
                    logging.warning(f"No tiled images found in {temp_path_tiled} for {file}. Skipping prediction.")
                    continue

                logging.info('Total number of tiles for {}: {}'.format(
                    os.path.basename(temp_path_data), len(testing_list_ids)))

                # get name, size, geo referencing data map
                logging.info('Extracting GeoData from Gridded data')
                testing_geoMap = io.getGeodata(testing_imageMap)

            except Exception as e:
                 logging.error(f"Failed to prepare data generator for {temp_path_data}: {e}")
                 continue # Skip to next file

            # Testing DataGenerator
            logging.info('Setting up Testing Generator')
            try:
                # Use config values consistently
                testing_generator = dataGenerator.DataGenerator(
                    list_IDs=testing_list_ids,
                    imageMap=testing_imageMap,
                    labelMap=None, # No labels for prediction
                    batch_size=config.batch,
                    n_classes=None, # Not needed for prediction input
                    image_channels=config.num_image_channels,
                    label_channels=None,
                    image_size=config.image_size, # Pass tuple directly
                    prediction=True, # Essential flag
                    shuffle=False) # Keep order for reconstruction
            except Exception as e:
                 logging.error(f"Failed to create DataGenerator for {temp_path_data}: {e}")
                 continue # Skip to next file

            # Get result
            logging.info(f"Starting prediction for {len(testing_list_ids)} tiles using {config.num_workers} workers...")
            pred_time = mtime.time()
            try:
                predictResult = unet_model.predict(
                    x=testing_generator,
                    workers=config.num_workers, # Use config value
                    use_multiprocessing=True, # This triggers the need for the fix
                    verbose=1
                )
                logging.info(f"Prediction finished in {mtime.time() - pred_time:.2f}s")

            except Exception as e:
                 # Catch potential OOM errors or other prediction issues
                 logging.error(f"Prediction failed for {temp_path_data}: {e}")
                 # Clean up potentially large variables if OOM is suspected
                 del testing_generator
                 del testing_list_ids
                 del testing_imageMap
                 del testing_geoMap
                 tf.keras.backend.clear_session() # Try to clear TF graph state
                 continue # Skip to next file

            logging.info('Predicted matrix shape for {}: {}'.format(
                os.path.basename(temp_path_data), predictResult.shape))

            logging.info('Saving individual tile predictions to: {}'.format(temp_path_predict))
            predict_save_time = mtime.time()
            # Iterating over predictions and saving it to geoReferenced TIF files
            temp_listPrediction = [] # List of paths to the predicted tiles
            try:
                for i in range(len(testing_list_ids)):
                    # Ensure index is valid
                    if i >= len(testing_geoMap) or i >= predictResult.shape[0]:
                        logging.warning(f"Index mismatch at {i}. GeoMap length: {len(testing_geoMap)}, Prediction length: {predictResult.shape[0]}")
                        continue

                    tile_id = testing_list_ids[i] # Use the ID for mapping
                    geo_info = testing_geoMap[i] # Get corresponding geo info
                    file_name = os.path.basename(geo_info['path']) # Get original tile filename

                    # Ensure prediction slice is valid
                    if predictResult[i, ...].size == 0:
                         logging.warning(f"Empty prediction slice for tile {file_name} (index {i}). Skipping.")
                         continue

                    # Process prediction: Apply threshold (e.g., 0.5) and convert to uint8
                    # Assuming single-channel output (binary segmentation)
                    labelPrediction = (predictResult[i, :, :, 0] > config.prediction_threshold).astype(np.uint8) * 255
                    # labelPrediction = np.round(predictResult[i, :, :, 0]).astype(np.uint8) * 255 # Alternative rounding

                    temp_path_output = os.path.join(temp_path_predict, file_name)

                    # Saving data to disk
                    io.write_tif(temp_path_output,
                                 labelPrediction, # Pass the processed 2D array
                                 geo_info['geoTransform'],
                                 geo_info['geoProjection'],
                                 (labelPrediction.shape[1], labelPrediction.shape[0])) # Use actual output size

                    temp_listPrediction.append(temp_path_output)

                logging.info(f"Saved {len(temp_listPrediction)} predicted tiles in {mtime.time() - predict_save_time:.2f}s")

            except Exception as e:
                logging.error(f"Error saving predicted tiles for {temp_path_data}: {e}")
                # Clean up potentially large prediction result
                del predictResult
                tf.keras.backend.clear_session()
                continue # Skip post-processing for this file

            # --- Cleanup prediction results from memory ---
            del predictResult
            del testing_generator # Release generator resources
            tf.keras.backend.clear_session() # Good practice after prediction loop


            # Merging Gridded dataset to single TIF
            merge_time = mtime.time()
            logging.info('Merging {} predicted tiles for {}. This may take a while'.format(
                len(temp_listPrediction), os.path.basename(temp_path_data)))

            # Define the final merged prediction output path
            temp_merged_output = os.path.join(path_merged_prediction, os.path.basename(file)) # Use original filename
            try:
                # Ensure the merge function handles potential errors and logs them
                io.mergeTile(listTIF=temp_listPrediction,
                             path_output=temp_merged_output,
                             method='gdal_merge') # Or specify method if needed
                logging.info(f"Merged prediction saved to {temp_merged_output} in {mtime.time() - merge_time:.2f}s")

            except Exception as e:
                 logging.error(f"Failed to merge tiles for {file}: {e}")
                 continue # Skip post-processing for this file


            # --- Post Processing ---
            postproc_st_time = mtime.time()
            logging.info(f"--- Starting Post-processing for {file} ---")

            # Define intermediate and final output paths using the base filename
            base_output_name = util.getNamenoExt(file)
            if args.upscalling:
                base_output_name += f"_{config.default_resolution}m"

            temp_erosion_output = os.path.join(path_erosion, f"{base_output_name}_eroded.tif")
            temp_watershed_output = os.path.join(path_watershed, f"{base_output_name}_watershed.tif")
            temp_vector_output_dir = os.path.join(path_vector, base_output_name) # Directory for vector outputs
            temp_simplify_output_dir = os.path.join(path_simplify, base_output_name) # Directory for simplified outputs
            temp_bbox_output_dir = os.path.join(path_bbox, base_output_name) # Directory for bbox outputs
            temp_skeletonize_output = os.path.join(path_merged_prediction, 'skeleton', f"{base_output_name}_skeleton.tif") # Skeleton path


            # Post Processing output image - Common first step: Erosion (often helps clean noise)
            logging.info(f'Post Processing: Erosion using filter size {filter}')
            erosion_time = mtime.time()
            try:
                postprocess.erosion(path_input=temp_merged_output,
                                    filter=filter,
                                    path_output=temp_erosion_output)
                logging.info(f'Erosion completed in {mtime.time() - erosion_time:.2f}s: {temp_erosion_output}')
                current_raster_input = temp_erosion_output # Input for next step
            except Exception as e:
                logging.error(f"Erosion failed for {file}: {e}")
                current_raster_input = temp_merged_output # Fallback to merged if erosion fails
                # Continue processing with the non-eroded raster? Or skip? Here we continue.


            if args.linearFeature is False: # Building Footprints or similar area features
                logging.info("Post Processing for Area Features (Watershed, Vectorization, Simplification, BBox)")

                # Watershed segmentation (optional, depends on if objects are touching)
                if config.use_watershed: # Add a flag in config.py
                    neighbour = config.watershed_neighbour
                    logging.info(f'Post Processing: Watershed Segmentation (neighbour={neighbour})')
                    ws_time = mtime.time()
                    try:
                        # Ensure input exists
                        if not os.path.exists(current_raster_input):
                             raise FileNotFoundError(f"Input for watershed not found: {current_raster_input}")

                        postprocess.watershedSegmentation(
                            current_raster_input, neighbour, temp_watershed_output)
                        logging.info(f'Watershed Segmentation completed in {mtime.time()-ws_time:.2f}s: {temp_watershed_output}')
                        current_raster_input = temp_watershed_output # Update input for vectorization
                    except Exception as e:
                        logging.error(f"Watershed Segmentation failed for {file}: {e}")
                        # Continue with the previous raster (eroded or merged) for vectorization
                else:
                     logging.info("Skipping Watershed Segmentation (config.use_watershed=False)")


                # Converting raster to Vector
                logging.info('Post Processing: Converting Raster to Vector (Polygon)')
                vec_time = mtime.time()
                util.check_dir(temp_vector_output_dir) # Ensure output dir exists
                try:
                     # Ensure input exists
                    if not os.path.exists(current_raster_input):
                        raise FileNotFoundError(f"Input for raster2vector not found: {current_raster_input}")

                    # raster2vector should return the path(s) to the created shapefile(s)
                    # Assuming it outputs to a specified dir, let's name the shapefile clearly
                    vector_shp_path = os.path.join(temp_vector_output_dir, f"{base_output_name}.shp")
                    io.raster2vector(path_raster=current_raster_input,
                                     path_output=vector_shp_path, # Pass full output path
                                     vector_type='polygon') # Specify polygon
                    logging.info(f'Raster to Vector (Polygon) completed in {mtime.time()-vec_time:.2f}s: {vector_shp_path}')
                    current_vector_input = vector_shp_path # Input for next step

                    # Simplification of polygons
                    logging.info(f'Post Processing: Simplifying Vector (Tolerance={config.simplify_parameter}m)')
                    simp_time = mtime.time()
                    util.check_dir(temp_simplify_output_dir) # Ensure output dir exists
                    temp_simplify_shp_path = os.path.join(temp_simplify_output_dir, f"{base_output_name}_simplified.shp")
                    try:
                         # Ensure input exists
                        if not os.path.exists(current_vector_input):
                             raise FileNotFoundError(f"Input for simplify_polygon not found: {current_vector_input}")

                        postprocess.simplify_polygon(path_shp=current_vector_input,
                                                     parameter=config.simplify_parameter,
                                                     path_output=temp_simplify_shp_path)
                        logging.info(f'Vector simplification completed in {mtime.time()-simp_time:.2f}s: {temp_simplify_shp_path}')
                        current_vector_input = temp_simplify_shp_path # Update input for bbox

                    except Exception as e:
                        logging.error(f"Vector simplification failed for {file}: {e}")
                        # Continue with the non-simplified vector for BBox? Or skip BBox? Here we continue.


                    # Shp to axis aligned bounding box
                    logging.info('Post Processing: Generating Axis-Aligned Bounding Box')
                    bbox_time = mtime.time()
                    util.check_dir(temp_bbox_output_dir)
                    temp_bbox_shp_path = os.path.join(temp_bbox_output_dir, f"{base_output_name}_bbox.shp")
                    try:
                        # Ensure input exists
                        if not os.path.exists(current_vector_input):
                             raise FileNotFoundError(f"Input for aabbox not found: {current_vector_input}")

                        postprocess.aabbox(path_shp=current_vector_input,
                                           path_output=temp_bbox_shp_path)
                        logging.info(f'AA BBox generation completed in {mtime.time()-bbox_time:.2f}s: {temp_bbox_shp_path}')

                    except Exception as e:
                        logging.error(f"AA BBox generation failed for {file}: {e}")

                except Exception as e:
                    logging.error(f"Raster to Vector (Polygon) failed for {file}: {e}")
                    # Cannot proceed with simplification or bbox if vectorization fails


            elif args.linearFeature is True: # Roads or similar linear features
                 logging.info("Post Processing for Linear Features (Skeletonization, Vectorization)")

                 # Skeletonization (thinning)
                 logging.info('Post Processing: Skeletonization')
                 skel_time = mtime.time()
                 util.check_dir(os.path.dirname(temp_skeletonize_output)) # Ensure dir exists
                 try:
                    # Ensure input exists
                    if not os.path.exists(current_raster_input): # Use the output of erosion
                        raise FileNotFoundError(f"Input for skeletonize not found: {current_raster_input}")

                    postprocess.skeletonize(
                        path_input=current_raster_input, # Input is typically the eroded binary mask
                        path_output=temp_skeletonize_output)
                    logging.info(f'Skeletonization completed in {mtime.time()-skel_time:.2f}s: {temp_skeletonize_output}')
                    current_raster_input = temp_skeletonize_output # Update for vectorization

                 except Exception as e:
                      logging.error(f"Skeletonization failed for {file}: {e}")
                      # Maybe try vectorizing the non-skeletonized version? Or skip? Here we skip vectorization.
                      continue # Skip vectorization if skeletonization fails


                 # Converting raster skeleton to Vector (LineString)
                 logging.info('Post Processing: Converting Raster to Vector (LineString)')
                 vec_time = mtime.time()
                 util.check_dir(temp_vector_output_dir) # Ensure output dir exists
                 try:
                    # Ensure input exists
                    if not os.path.exists(current_raster_input):
                        raise FileNotFoundError(f"Input for raster2vector (line) not found: {current_raster_input}")

                    vector_shp_path = os.path.join(temp_vector_output_dir, f"{base_output_name}_line.shp")
                    io.raster2vector(path_raster=current_raster_input,
                                     path_output=vector_shp_path, # Pass full output path
                                     vector_type='line') # Specify line
                    logging.info(f'Raster to Vector (LineString) completed in {mtime.time()-vec_time:.2f}s: {vector_shp_path}')

                 except Exception as e:
                    logging.error(f"Raster to Vector (LineString) failed for {file}: {e}")


            logging.info(f"--- Finished Post-processing for {file} in {mtime.time() - postproc_st_time:.2f}s ---")
            # --- End Post Processing ---

            # Record timing for this file
            timing[file] = mtime.time() - file_st_time
            logging.info(f"--- Total time for {file}: {timing[file]:.2f} seconds ---")

            # Clean up intermediate files for this image if desired
            # Be careful here - only delete if sure they are not needed later
            # Consider adding a --cleanup flag
            # e.g., os.remove(temp_merged_output), os.remove(temp_erosion_output), etc.
            # Also remove tiled and prediction folders for this file: shutil.rmtree(temp_path_tiled), shutil.rmtree(temp_path_predict)


    # --- End of main file loop ---

    # Saving overall timing to JSON
    total_time = mtime.time() - st_time
    timing['Total_Processing_Time'] = total_time
    timing_path = os.path.join(path_result, 'Timing.json')
    try:
        io.tojson(timing, timing_path)
        logging.info(f"Timing information saved to {timing_path}")
    except Exception as e:
        logging.error(f"Failed to save timing information: {e}")

    logging.info(f'=== Overall Process Completed in {total_time:.2f} seconds ===')

# --- Guard for multiprocessing ---
if __name__ == '__main__':
    multiprocessing.freeze_support()  # Essential for Windows packaged executables or spawn-based multiprocessing
    main() # Call the main function
    # sys.exit() # Optional: exit explicitly if needed, otherwise script exits naturally