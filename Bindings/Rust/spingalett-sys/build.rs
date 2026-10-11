// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

//! Links libspingalett: from SPINGALETT_LIB_DIR when set (a build tree's Bin/, an installed package's lib/),
//! otherwise from the linker's default paths (a system package, /usr/local/lib). Programs find the shared
//! library where the system's loader looks (LD_LIBRARY_PATH, PATH on Windows, an installed copy).

fn main() {
    println!("cargo:rerun-if-env-changed=SPINGALETT_LIB_DIR");
    if let Ok(dir) = std::env::var("SPINGALETT_LIB_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
    }
    println!("cargo:rustc-link-lib=dylib=spingalett");
}
