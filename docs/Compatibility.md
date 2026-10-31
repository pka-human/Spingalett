# Compatibility

What every 1.x release keeps.

From 1.0 on, Spingalett follows [semantic versioning](https://semver.org/): a 1.x release keeps
what 1.0 gave, and only 2.0 may take anything away.

- **Source:** a program written for 1.x compiles against every later 1.y. No function, type, struct
  field, enumerator or constant of the headers (`Spingalett.h`, `Spingalett.Runtime.h`,
  `Spingalett.Inference.h`, `Spingalett.Short.h`, `Spingalett.hpp`) is removed, renamed or given
  another meaning (a declaration may move to a header that `Spingalett.h` includes); releases add
  them. A new field takes the place of reserved words and means, at zero, what a program that does
  not know it expects, so the designated initializers of 1.0 keep their meaning.
- **Binary:** a program built against 1.x runs with every later 1.y without being built again: the
  shared library keeps the soname `libspingalett.so.1` (`libspingalett.1.dylib` on macOS), and the
  runtime `libspingalett-runtime.so.1` with the same symbol versions, the structs
  keep their size and the offsets of their fields, enumerators keep their values, and functions keep
  their parameters. On ELF platforms the functions carry symbol versions (`SPINGALETT_1.0`, then one
  per release that adds some), so that the loader refuses a library older than the program needs.
  The test `api.abi` compares every release with the one before.
- **Files:** every 1.x loads the `.slett` files 1.0 loads (formats 1 to 7; the engine runs 3 to 7;
  1.2 adds format 8 for transformers, 1.3 format 9 for the training features of language models, 1.4
  format 10 for GGUF's block precisions and today's attention, 1.5 format 11 for graphs of operators, whose
  operators' codes every later 1.x keeps)
  and `.slettd` files of formats 1 and 2, and writes the oldest format that holds what it saves; a
  later format only adds kinds of layers or coders, so a 1.0 engine runs every model of a later 1.x
  that uses only what it knows.
- **Python:** the `spingalett` package keeps its functions, classes and keyword arguments in the
  same way.
- **Not held:** the values of the `*_COUNT` enumerators and of `SPINGALETT_FORMAT_VERSION`, which
  grow with what a release adds; the text of messages; speed; and the last bits of results. Training
  repeats its bits on one version (on any number of threads, and on one GPU), but a release whose
  kernels add in another order may round differently. Everything outside `Include/` is internal.
- **Deprecation:** what is to go in 2.0 is marked deprecated in a 1.x release, in the header and the
  CHANGELOG, and keeps working until then.

The API (prefixed names, structs that can grow, zero as every field's default), the ABI (the soname `libspingalett.so.1`) and
the formats (`.slett` up to 12, `.slettd` 2) are what these rules keep. The network is an opaque handle, so its internal layout
can change without breaking programs; saved models are versioned and remain loadable.
