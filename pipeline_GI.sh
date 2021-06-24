#!/bash/bin


echo "-------------- SCRIPT TO EXECUTE GI PIPELINE -----------------"

# Assumed folder structure:
#   ~/base
#       /codes
#       /data
#           /images, /cleaned, /cropped, /points


dir=$(pwd)
echo "Current directory (should be codes folder):   "$dir

##########################
# ---- GI HISTOLOGY ---- #

echo "STEP 1: PROCESS THE IMAGE FILES"
python clean_crop.py
# output:   (i) Full cleaned images: ~/data/cleaned/{ID}/{file}.png
#           (ii) Cropped images: ~/data/cropped/{train,test}/{ID}/{tissue}/{file}.png
#           (iii) Crop location ~/data/dat_idx_crops.csv

echo "STEP 2: PROCESSING ROBARTS"
python lbl_robarts.py
# output: ../data/df_lbls_robarts.csv

echo "STEP 3: PROCESSING NANCY"
python lbl_nancy.py
# output: ../data/df_lbls_nancy.csv

echo "STEP 4: ANONYMIZE DATA"
python data_anonymize.py
# output: ../data/df_lbls_anon.csv
#         ../data/df_codebreaker.csv
#         ../data/cropped/{train/test}/{ID}/*.png

echo "STEP 5: TRAINING ORDINAL MODEL"
python cnn_ordinal_nancy_robarts.py
# output: ../data/di_ID.npy
#         /saved_networks/cnn_conc_epoch{#}.pt

echo "STEP 6: EVALUATE MODEL"
python eval_ordinal.py
# output: ../data/df_ordinal_score.csv

############################
# ---- CELL ANNOTATOR ---- #

echo "STEP 1: SELECT PATIENTS/IMAGES FOR TRAINING"
python patient_select.py
# output: ../data/cell_counter/*.png

echo "STEP 2: RUN SCRIPTS ON ANNOTATION POINTS"
python qupath_points.py






