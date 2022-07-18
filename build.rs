/* -----------------------------------------------------------------------------
	MithrilEngine Framework (MEF)

	Copyright (c) 2021-2022, daigennki (@daigennki)
----------------------------------------------------------------------------- */
use std::io::Read;
use std::path::Path;
use std::ffi::OsStr;

fn main()
{
	// Compile HLSL shaders in src/shaders into SPIR-V.
	
	// `cd` into the shader source folder so that `#include` in shader source files work correctly.
	std::env::set_current_dir(Path::new("src/shaders")).unwrap();

	let shader_paths = std::fs::read_dir("./").unwrap()
		.map(|d| d.unwrap().path())
		.filter(|p| p.extension().unwrap_or(OsStr::new("")) == "hlsl");

	for shader_path in shader_paths {
		match detect_shader_model(&shader_path) {
			Some(shader_model) => {
				let mut hlsl_file = std::fs::File::open(&shader_path).unwrap();
				let mut hlsl_src_str = String::new();
				hlsl_file.read_to_string(&mut hlsl_src_str).unwrap();

				let shader_file_name = shader_path.file_name().unwrap().to_str().unwrap();

				let spirv = hassle_rs::compile_hlsl(
					shader_file_name,
					&hlsl_src_str,
					"main",
					&shader_model,
					&vec!["-spirv"],
					&[]
				).unwrap();

				let shader_file_stem = shader_path.file_stem().unwrap().to_str().unwrap();
				
				let output_path = format!("../../shaders/{}.spv", shader_file_stem);
				std::fs::write(&output_path, spirv).unwrap();
				println!("wrote {}", output_path);
			},
			None => println!("could not detect shader model for shader '{}', skipping...", shader_path.to_str().unwrap())
		}
	}

	std::env::set_current_dir(std::path::Path::new("../../")).unwrap();

	// This will only run if files in the src/shaders directory have been changed,
	// or if this build script has been changed.
	println!("cargo:rerun-if-changed=src/shaders/");
	println!("cargo:rerun-if-changed=build.rs");
}

fn detect_shader_model(path: &std::path::Path) -> Option<&str>
{
	let path_str = path.to_str().unwrap();
	if path_str.ends_with(".vert.hlsl") {
		Some("vs_6_5")
	} else if path_str.ends_with(".frag.hlsl") {
		Some("ps_6_5")	
	} else {
		None
	}
}

