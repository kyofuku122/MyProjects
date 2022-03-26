# Preparing Data

#### <ins>Prepare Directory</ins>

1. Create directory like below

   ```
   tlt-experiments
   ├── smartphone_data
   │  ├── tfrecords
   │  ├── training
   │  │  ├── image　# images of smartphone
   │  │  │   ├── 1.jpg
   │  │  │   ├── 2.jpg
   │  │  │   ├── 3.jpg
   │  │  │   ├── …
   │  │  │   └── 30.png
   │  │  └── label #Directory for kitti data(which we will create on the next step)
   │  │      ├── 1.txt
   │  │      ├── 2.txt
   │  │      ├── 3.txt
   │  │      ├── …
   │  │      └── 30.txt
   │  ├── testing
   │  │  └── image
   │  │      ├── 1.jpg
   │  │      ├── 2.jpg
   │  │      ├── 3.jpg
   │  │      ├── …
   │  │      └── 30.png

   ```

   <br>

#### <ins>Prepare Kitti Dataset</ins>

1. Annotate Data with XML format <br>
   (In case you don't know how, maybe you can check out my tutorial on that([BLOG:Tutorial](https://kyosukefukumoto.blog/54-annotating-images-with-labelimg/))
1. Use [_xml2kitti.py_](./xml2kitti.py) to convert your xml files to kitti data format

   ```shell
   python xml2kitty.py /PATH/TO/XML/DIRECTORY
   ```

1. Place generated kitti data to the directory above

<br>

#### <ins>Convert kitti data to _tfrecord_</ins>

1. Use [_detectnet_tfrecords_kitti_trainval.txt_](./detectnet_tfrecords_kitti_trainval.txt) to convert kitti data to _tfrecord_<br>
   (Run inside docker container)

   ```shell
   tlt-dataset-convert -d ./detectnet_tfrecords_kitti_trainval.txt \
   -o /workspace/tlt-experiments/smartphone_data/tfrecords/kitti_trainval/kitti_trainval
   ```

   ※_detectnet_tfrecords_kitti_trainval.txt_

   ```
   kitti_config {
   root_directory_path: "/workspace/tlt-experiments/smartphone_data/training"
   image_dir_name: "image"
   label_dir_name: "label"
   image_extension: ".jpg"
   partition_mode: "random"
   num_partitions: 2

   val_split: 34
   # Split 34% of the data as validation set

   num_shards: 10
   }
   image_directory_path: "/workspace/tlt-experiments/smartphone_data/training"

   ```
