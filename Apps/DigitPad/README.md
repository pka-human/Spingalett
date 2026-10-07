# DigitPad

A small desktop app built on Spingalett: draw a digit with the mouse and a network trained on MNIST
classifies it while you draw. It shows the predicted digit, the probability of every class and the
28x28 image the network actually receives.

![DigitPad recognizing a hand-drawn 3](screenshot.png)

The model is a convolutional network with batch normalization: two stages of two 3x3
convolutions (32, then 64 filters), each followed by batch normalization and ReLU, with 2x2 max
pooling after each stage, then a dense layer of 128 units (batch normalization, ReLU, dropout 0.3)
and a softmax output; 468K parameters, trained with AdamW and a warm-up cosine schedule. It
reaches **99.58%** accuracy on the MNIST test set and 99.45% on a randomly distorted copy of it
(the 784-1024-512-10 MLP of earlier releases: 99.27% and 98.44%).

Every [release](https://github.com/pka-human/Spingalett/releases) has a ready-to-run build for
Linux and one for Windows.

## Running the AppImage

```bash
chmod +x DigitPad-*-x86_64.AppImage
./DigitPad-*-x86_64.AppImage
```

The image contains the app, `libspingalett`, a static SDL2 and the model. It needs an x86-64
Linux with glibc 2.34 or newer (Ubuntu 22.04, Debian 12, Fedora 35, RHEL 9 or later) and an X11
or Wayland session; display and OpenGL libraries are taken from the system. Without FUSE, run it
with `--appimage-extract-and-run`.

## Running on Windows

Unpack `DigitPad-<version>-windows-x86_64.zip` anywhere and start `DigitPad.exe` (64-bit
Windows 10 or 11). The folder holds the program, the model, `libspingalett.dll` and `SDL2.dll`,
with a `README.txt` and the licences. The program is not signed,
so SmartScreen may stop it the first time: **More info**, then **Run anyway**.

| Input | Action |
|---|---|
| Left mouse button | draw |
| Right mouse button | erase |
| `C`, `Space`, `Backspace` or **Clear** | clear the canvas |
| `Ctrl+Z`, `U` or **Undo** | undo the last stroke (32 levels) |
| `Esc` | quit |

Command-line options:

```
DigitPad [--verbose] [model.slett]            open the window; --verbose prints each prediction
DigitPad --classify image.pgm [model.slett]   classify a binary PGM image and exit
```

Without a model argument the app uses `$DIGITPAD_MODEL`, then `../share/digitpad/mnist.slett`
relative to the executable, then `mnist.slett` next to it. On Windows it is a GUI program that
prints to the console it was started from; `DigitPad.exe --classify image.pgm | more` waits for
the result in `cmd`.

## How a drawing becomes an input

MNIST digits were made by fitting each digit into a 20x20 box, keeping its aspect ratio, and
centering its center of mass in a 28x28 field. `digit_normalize()` in [Digits.h](Digits.h) does the
same with a drawing: it finds the bounding box of the ink, resamples it into 20x20 with area
averaging (a summed-area table, so the result is anti-aliased like the scans), and shifts it so
the center of mass lands at (14, 14). Drawings of any size and position therefore look alike to
the network.

Digits drawn with a mouse still differ from scanned handwriting, so the trainer augments every
training sample on the fly: random rotation (up to 15 degrees), scale (0.8 to 1.15), aspect ratio,
shear, shift (up to 2.5 pixels) and stroke thickness (dilation or erosion). The samples come from a
`MODE_GENERATOR_FUNCTION` data generator, so each epoch sees new variants of the 55,000 training
images.

## Building

```bash
# library, app and trainer (the app needs the SDL2 development package, e.g. libsdl2-dev)
cmake -S . -B Build -DCMAKE_BUILD_TYPE=Release -DBUILD_APPS=ON -DBUILD_WITH_OPENMP=ON
cmake --build Build --parallel

# train a model: about 19 minutes for 30 epochs with OpenMP on a 4-core machine
sh Examples/download_mnist.sh data/mnist
Bin/DigitPadTrain data/mnist mnist.slett 30

Bin/DigitPad mnist.slett
```

`DigitPadTrain` keeps the last 5,000 training images for validation, saves the epoch with the best
validation accuracy and finally reports test accuracy, clean and distorted. It also writes
`mnist.slett.info`, the one-line description shown in the app's footer.

## Building the AppImage

```bash
Apps/DigitPad/Package/build-appimage.sh                    # downloads MNIST and trains a model
Apps/DigitPad/Package/build-appimage.sh --model mnist.slett   # packages an existing model
```

The script builds SDL2 from source as a static library with only its video subsystem (X11,
Wayland and OpenGL are loaded at run time), builds Spingalett for baseline x86-64 without OpenMP,
assembles the AppDir and packs it with `appimagetool`, which it downloads if it is not installed.
The result is written to `build/appimage/`.

## Building the Windows zip

```bash
# in an MSYS2 UCRT64 shell
pacman -S mingw-w64-ucrt-x86_64-{gcc,cmake,ninja,SDL2}
Apps/DigitPad/Package/build-windows-zip.sh --model mnist.slett

# or on Linux, cross-compiling with MinGW-w64 (apt-get install mingw-w64); SDL2 is built from source
Apps/DigitPad/Package/build-windows-zip.sh --model mnist.slett
```

Without `--model` the script downloads MNIST and trains a model first, as the AppImage script
does. It builds DigitPad for baseline x86-64 without OpenMP, copies every DLL the program loads
that is not part of Windows (found by reading the import tables, so nothing is left out), adds
the model, the licences and a `README.txt`, and writes `build/windows/DigitPad-<version>-windows-x86_64.zip`.
`DigitPad.exe` carries an icon, version information and a manifest that makes UTF-8 the process
code page, so the model loads from folders whose names are not ASCII.
`Package/test-windows-zip.ps1` is the check the release workflow runs on Windows: it classifies
`seven.pgm` from the command line, then opens the window, draws a 7 with the mouse and expects
DigitPad to recognise it.

## Files

| File | Contents |
|---|---|
| `DigitPad.c` | the app: software-rendered UI on an SDL2 window |
| `Train.c` | the trainer |
| `Digits.h` | normalization and augmentation shared by both |
| `FontAtlas.h` | DejaVu Sans glyphs, generated by `Tools/make_font_atlas.py` |
| `Package/` | AppImage and Windows zip scripts, desktop entry, icons, Windows resources, the Windows test |

The interface text is rendered from `FontAtlas.h`, so the app needs no font library. The glyphs
come from the DejaVu fonts ([license](https://dejavu-fonts.github.io/License.html)).
