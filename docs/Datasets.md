# Data sets

Reading data into memory, and `.slettd` files, which are compact and stream into training. [DatasetFormat.md](DatasetFormat.md) specifies the format.

## Data sets

```c
SpingalettDataset train_set, val_set;
spingalett_load_idx("train-images-idx3-ubyte", "train-labels-idx1-ubyte", 10, &train_set);
spingalett_dataset_shuffle(&train_set);                 /* optional, uses spingalett_seed() */
spingalett_dataset_split(&train_set, 5000, &val_set);   /* last 5,000 samples -> val_set */
/* ... train(.inputs = train_set.inputs, .targets = train_set.targets,
                        .sample_count = train_set.count, ...) */
spingalett_dataset_free(&train_set);
spingalett_dataset_free(&val_set);
```

`spingalett_load_idx()` reads the IDX format used by MNIST: unsigned-byte samples of any shape
are flattened and scaled to [0, 1] (float and double files are read unchanged), and labels are
one-hot encoded. `spingalett_load_csv(path, target_columns, num_classes, &d)` reads numeric,
comma-separated files; a non-numeric first line is skipped as a header, the last `target_columns`
columns are the targets, and with `num_classes > 0` a single label column is one-hot encoded.
`spingalett_load_cifar(paths, count, num_classes, &d)` reads CIFAR binary batches one after the
other as 32 x 32 x 3 channels-last images in [0, 1]: `num_classes` 10 for CIFAR-10, 100 or 20 for
the fine or coarse labels of CIFAR-100.

## Data set files

`.slettd` is Spingalett's own data set format: binary, compact and loaded straight into a
`SpingalettDataset`. By default every stream is stored in the smallest encoding that keeps all
values exact (8-bit `q / 255` for image data, IEEE half, or float32; one-hot targets as class
indices) and compressed with adaptive context models that learn which earlier values predict the
next, such as the pixel above in an image: a binary range coder, or for dense data such as
photographs an rANS coder of half-bytes that decodes two to three times as fast. Lossy FP16, BF16
and per-feature 8-bit encodings are available on request. Files consist of independently
decodable chunks with CRC-32 checksums, and can record the input shape, class names and further
sets of targets for the same samples (CIFAR-100's fine and coarse labels, for example).

```c
spingalett_save_dataset(&train_set, "mnist-train", NULL);         /* writes mnist-train.slettd */
SpingalettDataset d = {0};
spingalett_load_dataset("mnist-train.slettd", &d);                 /* bit-identical to train_set */

SpingalettDatasetReader *r = spingalett_dataset_open("mnist-train.slettd", true);   /* shuffled */
train(.net = net, .training_mode = MODE_GENERATOR_FUNCTION,
      .generator = spingalett_dataset_generator, .generator_data = r, .epochs = 10,
      .training_strategy = STRATEGY_SMALL_BATCH, .batch_size = 128);
spingalett_dataset_close(r);
```

A file can be used in three ways, chosen by its size against the memory at hand:

| Way | Memory | Use it when |
|---|---|---|
| `spingalett_load_dataset()`, then `spingalett_train()` on the arrays | 4 bytes per value | the data fits in memory as float |
| a reader with `.in_memory = true` (`spingalett_dataset_open_ex`) | 1 byte per 8-bit value, 2 per FP16 | it fits in its compact form; passes shuffle all samples, as with arrays |
| a streaming reader (`spingalett_dataset_open`) | a few chunks of about 1 MB | it does not fit; passes shuffle the chunks, and the samples within each chunk |

Loading decodes the chunks in parallel with OpenMP. Readers convert values to float a batch at a
time. A streaming reader decodes the next chunks on a background thread when a processor is free
for it, and otherwise decodes several chunks at a time on the OpenMP threads when it needs them,
since a thread competing with the OpenMP threads for the processors would stall them; the samples
come in the same order either way: a pass's order is a function of the reader's seed and the pass
number. `spingalett_dataset_get_position()` gives where a reader stands (its seed, its pass and the
samples of the pass already read), and `spingalett_dataset_set_position()` moves a reader of the same
file and options there, after which it reads what the first would have read, bit for bit. With
`spingalett_dataset_open_u8()`, 8-bit data already in
memory trains through a reader without a float copy. One epoch of a small CNN on CIFAR-10 (two
convolutions, 4 threads, Xeon @ 2.1 GHz, 4 vCPUs, medians):

| Source of the 50,000 training images | Memory | Time per epoch |
|---|---:|---:|
| float arrays | 616 MB | 4.5 s |
| `.slettd` in memory (8-bit) | 154 MB | 4.6 s |
| `.slettd` streamed | about 10 MB | 5.5 s (0.8: 14.7 s) |

Saving is parallel too: chunks are compressed side by side and written in order (CIFAR-10's
training set: 2.2 s on 4 threads, loading 1.1 s against 2.7 s in 0.8).

| Training set | float32 | Original files | `gzip -9` / `xz -9` of them | `.slettd` (lossless) |
|---|---:|---:|---:|---:|
| MNIST (60,000 images and labels) | 190.6 MB | 47.1 MB (IDX) | 9.7 / 7.9 MB (images only) | 7.8 MB |
| CIFAR-10 (50,000 images and labels) | 616.4 MB | 153.7 MB (binary batches) | 141.7 / 116.3 MB | 112.1 MB |

The shape and class names travel with a `SpingalettDataset` (`height`, `width`, `channels`,
`class_names`, filled by the IDX and CIFAR readers and by `.slettd` files that record them; set
names with `spingalett_dataset_set_class_names()`). Further sets of targets are saved through
`SpingalettDatasetSaveOptions.extra_targets` and loaded with
`spingalett_load_dataset_targets(path, set, &d)` or `SpingalettDatasetReaderOptions.target_set`;
`spingalett_dataset_info()` and `spingalett_dataset_class_name()` describe a reader's file.
`spingalett_load_dataset_from_memory()` reads a file image already in memory.

`Bin/DatasetTool` converts data sets and inspects files:

```sh
DatasetTool idx train-images-idx3-ubyte train-labels-idx1-ubyte mnist-train.slettd
DatasetTool csv data.csv 1 3 iris.slettd             # last column: a label of 3 classes
DatasetTool cifar cifar10-train.slettd data_batch_{1,2,3,4,5}.bin
DatasetTool cifar cifar100-train.slettd train.bin --cifar100      # fine and coarse labels
DatasetTool images photos/ photos.slettd --size 64x64 --rgb       # one subfolder per class
DatasetTool info cifar10-train.slettd                # encodings, shape, sets of targets, class names
DatasetTool verify cifar10-train.slettd              # decodes every chunk, checks every checksum
```

`cifar` reads the class names from `batches.meta.txt` (CIFAR-10) or `fine_label_names.txt` and
`coarse_label_names.txt` (CIFAR-100) next to the batches when they are there. `images` reads PNG,
JPEG, BMP, TGA, GIF, PSD, HDR, PIC and PNM files (with stb_image), names the classes after the
subfolders, and resizes by area averaging. The format is specified in
[docs/DatasetFormat.md](DatasetFormat.md).
