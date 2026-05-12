fn main() {
    #[cfg(feature = "bench-vs-c")]
    {
        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:rerun-if-changed=cprobe/libsais_probe.c");
        println!("cargo:rerun-if-changed=libsais/src/libsais.c");
        println!("cargo:rerun-if-changed=libsais/include/libsais.h");

        cc::Build::new()
            .file("cprobe/libsais_probe.c")
            .include("libsais/include")
            .flag_if_supported("-std=c99")
            .compile("libsais_probe");
    }
}
